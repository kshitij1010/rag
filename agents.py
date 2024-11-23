from __future__ import annotations
from pathlib import Path
from openai import OpenAI
from openai.types import chat

import json
import abc
import logging
from rich.logging import RichHandler
import os
from typing import Collection, List

import dialogue
from dialogue import Dialogue
import characters
from characters import Character
from tracking import default_client, default_model, default_eval_model


# Set up logging.  For usage see findsim.py in earlier assignments.
log = logging.getLogger(Path(__file__).stem)    
if not log.hasHandlers():   # avoid adding again when reimporting
    log.addHandler(RichHandler(level="NOTSET", markup=True,   # allows rich text log messages, with [color]s
                               show_time=False, show_level=False))
log.setLevel(logging.WARNING)   # usually WARNING, but you may want to change to INFO or DEBUG to get more output


class Agent:
    """An AI agent whose actions consist of adding turns to dialogues.
    
    (If you've studied AI or reinforcement learning, regard the dialogue so far
    as the state of the environment.  As usual, this state when the agent takes
    an action, both because the agent extends the dialogue and because the
    environment responds by extending it further (when the other speaker talks).
    The evaluation scores can also be viewed as rewards given by the environment.)
     
    This class must be subclassed to specify a *particular* policy for choosing
    the next action.  This could involve a strategy of calling the LLM, one or
    more times."""
    
    name: str = "Override me!"

    # This is the main method that Agents must implement!
    @abc.abstractmethod
    def response(self, d: Dialogue, **kwargs) -> str:
        """Generate the next turn and return it.  This corresponds to choosing
        an action from the policy.  How this is done depends on the specific
        agent subclass and instance, and may use randomness."""
        raise NotImplementedError("Override me!")

    # Everything else is convenience methods based on `response()`.
    def respond(self, d: Dialogue, **kwargs) -> Dialogue:
        """Generate the next turn and add it nondestructively to the dialogue.
        This corresponds to choosing and carrying out the action."""
        return d.add(self.name, self.response(d), **kwargs)
    
    def ask(self, d: Dialogue, 
            speaker: str, question: str, **kwargs) -> Dialogue:
        """Nondestructively extend the dialogue with the given
        turn and the agent's response to it."""
        return self.respond(d.add(speaker, question), **kwargs)
    
    def ask_quietly(self, d: Dialogue, 
            speaker: str, question: str, **kwargs) -> str:
        """Like `ask`, but only return the response, not an
        extended dialogue.  This is useful for asking questions
        without giving the agent a memory of them."""
        return self.response(d.add(speaker, question), **kwargs)
           
    def converse(self, 
                 prefix: Dialogue = Dialogue(),   # default to empty dialogue
                 username: str = os.environ.get("USERNAME") or os.environ.get("USER") or "Human User",
                 userfirst: bool = True
                 ) -> Dialogue:
        """Create or extend a dialogue by talking to the Python user.
        By default, the user goes first.
        The user can give a blank response to end the dialogue."""
        
        d = prefix    # build up the dialogue here
        if not userfirst:
            d = self.respond(d)  # agent starts

        # It's now the user's first turn
        print(d, flush=True)    # immediately show the user the dialogue so far
        
        while True:
            # get the user's turn
            content = input(f"Say something to {self.name}: ")
            if content.strip() == "":
                return d
            d = d.add(username, content)
            
            # Now add the system's turn, and immediately show both new turns
            d = self.respond(d)
            print(d[-2:], flush=True)
        return d
    

class ConstantAgent(Agent):
    """A conversational agent that always says the same thing."""

    def __init__(self, name: str, response: str) -> None:
        self.name = name
        self.response_str = response

    def response(self, d: Dialogue, **kwargs) -> str:
        return self.response_str
    

class LLMAgent(Agent):
    """A conversational agent that uses an LLM to respond.
    This may be subclassed."""

    def __init__(self,
                 name: str,
                 model: str = default_model,      # allow overriding the default model
                 client: OpenAI = default_client, # allow overriding the default client
                 **kwargs                         # for both dialogue.format_as_messages and openai.chat.completions.create 
                ) -> None:
        self.name = name
        self.model = model
        self.client = client

        # Split the kwargs dictionary into two dictionaries.
        kws_format = ['system', 'system_last', 'speaker_names', 'compress', 'tool', 'tool_name']
        self.kwargs_format = {kw: kwargs[kw] for kw in kwargs if kw in kws_format}      # e.g., system
        self.kwargs_llm    = {kw: kwargs[kw] for kw in kwargs if kw not in kws_format}  # e.g., temperature
        
    def __repr__(self) -> str:
       return f"<LLMAgent {self.name}>"
        
    def response(self, d: Dialogue, **kwargs) -> str:
        
        # Just ask the LLM to continue the dialogue.
        
        messages = dialogue.format_as_messages(d, speaker=self.name, **self.kwargs_format)                                           
        
        pretty_messages = '\n'.join([f"[black on bright_yellow]({m['role']})"
                                     f"[/black on bright_yellow] {m['content']}" for m in messages])
        log.info(f"Calling LLM:\n{pretty_messages}")
        ##### NEXT LINE IS WHERE THE MAGIC HAPPENS #####
        response = self.client.chat.completions.create(messages=messages, 
                                                       model=self.model, **self.kwargs_llm)  
        log.debug(f"Response from LLM:\n[black on white]{response}[/black on white]")

        # That's it - now we have our response!  Get the content out of it.
        
        choice: chat.chat_completion.Choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str):
            raise ValueError("No content string returned from {self.kwargs_llm['client']}")

        # Clean up the returned content a little bit.
        
        if choice.finish_reason == 'length':
            # indicate that response was cut off due to max_tokens
            content += " ..."

        speaker = f"{self.name}: "
        if content.startswith(speaker):
            # Generated response was unfortunately in the form "Alice: I agree with you." 
            # Remove the "Alice: " part.
            # (This could happen if the messages you sent to the LLM included speaker names,
            # for example if you called `format_as_messages` with speaker_names=True.)
            content = content[len(speaker):]  
            
        # Log the content part of the LLM's response, but only if
        # we didn't already log the whole thing above.
        if log.getEffectiveLevel() > logging.DEBUG:
            log.info(f"Response from LLM:\n[black on white]{content}[/black on white]")

        return content

# Utility function that can be useful in making prompts.
def conjunction(items: Collection, 
                conj: str = "and", oxford: bool = True, 
                zeroval: str | None = None) -> str:
    """Combines items into a single string, using a linguistic conjunction
    such as "and" or "or".  If there are no items, raise an exception, or
    return `zeroval` if defined."""
    strs: List[str] = [str(x) for x in items]  # if items was unordered, this imposes an order
    if len(strs) == 0:
        if zeroval is None:
            raise ValueError("Can't conjoin 0 items")
        return zeroval
    elif len(strs) == 1: 
        return strs[0]
    else:
        conj = " " + conj.lstrip()  # one leading space
        if len(strs) > 2 and oxford:
            conj = ","+conj   # precede that space with a comma
        return ", ".join(strs[:-1]) + conj + " " + strs[-1]

class CharacterAgent(LLMAgent):
    """An LLM agent that simulates how a specific Character would converse.
    
    We would prefer to test our argubots by having actual humans talk to them,
    but it's quicker to have CharacterAgents do the testing instead."""

    def __init__(self, 
                 character: Character,
                 name: str | None = None,   # allow overriding the character's name
                 **kwargs
                ) -> None:
    
        # Derive a name and system prompt from the character data.
        if name is None: name = character.name   # by default, name the agent after the character         
        if character.languages:
            langprefs = f", and you prefer to speak {conjunction(character.languages, conj='or')}"
        else:
            langprefs = ""
        system = (f"Your name is {character.name}{langprefs}. "
                  f"You are {character.persona}. "
                  f"{character.conversational_style}"
                  f"\n\nReply in 1 sentence. Don't repeat your previous points.")

        # Now initialize self.
        super().__init__(name, system=system, **kwargs)
        self.character = character
        self.conversation_starters = character.conversation_starters   # can be examined by simulate.simulated_dialogue
        
    def __repr__(self) -> str:
        if self.name == self.character.name:
            return f"<CharacterAgent for character {self.name}>"
        else:
            return f"<CharacterAgent {self.name} for character {self.character.name}>"


class EvaluationAgent(LLMAgent):
    """An agent that is designed to answer our questions about a piece of text,
    namely a dialogue script. The agent evaluates from the viewpoint of a
    particular Character, which may itself be mentioned in the script.
  
    This is in lieu of asking human participants or obervers (e.g., political
    science grad students) to rate the texts.

    While a CharacterAgent and an EvaluationAgent may both be based on the same
    underlying Character, they will act differently.  A CharacterAgent is
    carrying on a political conversation, using its languages and conversational
    style.  An EvaluationAgent is just reporting its private thoughts to a third
    party."""  
  
    def __init__(self, 
                 character: Character,
                 **kwargs,
                ) -> None:

        # Derive a name and system prompt from the character data.
        name = f"{character.name} as evaluator"
        system = (f"Your name is {character.name} and you are {character.persona}."
                  f"\n\nThe user will show you a conversation and ask you a few "
                  f"questions about it. Answer them concisely and honestly.")

        # Now initialize self.
        super().__init__(name, system=system, **kwargs)
        self.character = character
      
    def __repr__(self) -> str:
        return f"<EvaluationAgent for character {self.character.name}>"
    
    def response(self, d: Dialogue, **kwargs) -> str:
        # Unless kwargs specifies otherwise, use temperature=0 and the eval model.
        kwargs = { 'temperature': 0, 'model': default_eval_model } | kwargs
        return super().response(d, **kwargs)
    
    def rating(self, d: Dialogue,
               speaker: str, question: str,
               lo: int, hi: int) -> int | None:
        """Like `ask_quietly()`, but try to get an integer in the given range.
        
        Raise a ValueError if the generated response isn't a number.
        (The caller could try again, perhaps with a different `question` or 
        a higher `temperature`, but there's no guarantee of ever succeeding!)
        """

        # TODO: In principle, we should do constrained decoding here, for example, by
        # using the API to boost the logits of numbers in this range (or their tokens).
        # Another option would be to generate with temperature=1, n=20, and average over
        # the numeric elements of the 20 return values.
        
        s = self.ask_quietly(d, speaker,
                             question
                             + f"\n\nReply with a single integer in the range {lo}-{hi}. Say nothing else.")
        i = int(s)  # will throw a ValueError if s doesn't look like an int
        if not lo <= i <= hi:
            raise ValueError("LLM agent generated rating {i} that's out of range [{lo}, {hi}]")
        return i

##############################
# Define some CharacterAgents.
##############################

# Define agents.devset based on characters.devset
# Each Character has a corresponding simple CharacterAgent.
devset = [CharacterAgent(char) for char in characters.devset]

# Also, for convenience in the notebook: if characters.bob is a Character, 
# then we'll define agents.bob as a corresponding CharacterAgent.
# So, we'll loop through all of the Characters defined at the top level of the characters module.

for name, value in vars(characters).items():    # top-level attributes of the chars module
    if isinstance(value, Character):                # if it's a top-level var bound to a Character 
        char: Character = value                         # get that Character
        agent = CharacterAgent(char)                    # make a CharacterAgent (might duplicate one from devset)
        vars()[name] = agent                            # assign it to a top-level var in current module
