from __future__ import annotations
from pathlib import Path
import logging
from rich.logging import RichHandler
from typing import Counter, DefaultDict, List, Tuple
from collections import Counter, defaultdict
import itertools

from agents import Agent, CharacterAgent, EvaluationAgent, conjunction
from characters import Character, devset as dev_chars
from dialogue import Dialogue
from simulate import simulated_dialogue
from tracking import read_usage

log = logging.getLogger(Path(__file__).stem)    
if not log.hasHandlers():   # avoid adding again when reimporting
    log.addHandler(RichHandler(level="NOTSET", markup=True,   # allows rich text log messages, with [color]s
                               show_time=False, show_level=False))
log.setLevel(logging.WARNING)   # usually WARNING, but you may want to change to INFO or DEBUG to get more output


# A couple of useful constants used later in the file.

research_team = "NLP class students"  # name of the speaker who is asking the evaluation questions
judge = Character("Judge Wise", [],   # the external observer who is answering some of those questions
                  "a social scientist who studies and assesses political conversations")   

# We're going to make a lot of evaluations, so we'd better have a convenient way
# to represent and aggregate them.

class Eval():
    """Aggregated results from one or more dialogue evaluations.

    We track the mean score on each numerical question (ignoring missing values),
    and the list of long-form comments for each free-form question.
    
    This class is boring from an NLP point of view -- just a utility class.
    But it is reasonably general; it could handle other questions.
    """
    scores: Counter[str]     # total score on each criterion
    denoms: Counter[str]     # number of scores contributing to the total, for each criterion   
    comments: DefaultDict[str, List[Tuple[str,str]]]  # list of (evaluator,comment) pairs for each long-form question

    def __init__(self,
                 comments: dict[str,List[Tuple[str,str]]] = {},    
                 scores: dict[str,int] | None = None,
                 denoms: dict[str,int] | None = None,
                 n: int = 1        
                ) -> None: 
        self.comments = defaultdict(list)
        for key, val in comments.items():
            self.comments[key] = val

        self.scores = Counter(scores)

        if denoms is None:
            denoms = {key: 1 for key in self.scores}
        self.denoms = Counter(denoms)
        if set(self.scores.keys()) != set(self.denoms.keys()):
            raise ValueError(f"scores and denoms have different sets of keys: {scores}, {denoms}")
            
    def mean(self) -> dict[str,float]:
        m = {k: self.scores[k]/self.denoms[k] for k in self.scores}
        m['TOTAL'] = sum(m.values())
        return m
            
    def __repr__(self) -> str:         
        count = max(self.denoms.values())
        
        allcomments = [f"Comments from {question} question:\n"
                        + '\n'.join(f"({c[0]}) {c[1]}" for c in commentlist)
                        for question, commentlist in self.comments.items()]
        
        return (f"<Eval of ≈ {count} dialogues:\n{repr(self.mean())}\n\n"
                + '\n\n'.join(allcomments))
        
    def __iadd__(self, other: Eval) -> Eval:   # the += operator
        if not isinstance(other, Eval):
            raise ValueError(f"Can only add Evals to Evals, but got {type(other)}")
        self.scores += other.scores     # sum Counter dictionaries
        self.denoms += other.denoms
        for key, val in other.comments.items():
            self.comments[key] += val   # destructively append lists
        return self

    def __add__(self, other: Eval) -> Eval:   # the + operator
        if not isinstance(other, Eval):
            raise ValueError(f"Can only add Evals to Evals, but got {type(other)}")
        comments = defaultdict(list)  # collect all comments here
        for key, val in itertools.chain(self.comments.items(), other.comments.items()):
            comments[key] += val   # append lists
        return Eval(comments,
                    self.scores + other.scores,
                    self.denoms + other.denoms)


# The prompt text is hardcoded into the two top-level functions below and the
# EvaluationAgent class.
#
# That's easiest to understand, and it's okay for this assignment, because the
# evaluation metric is fixed and you don't need any flexibilty to change it.
# 
# But if you were trying to engineer the evaluation scheme to agree with real
# human evaluations, you would want to create many different evaluation objects
# and loop over them.

def eval_by_participant(participant: Character,
                        other: str, dialogue: Dialogue) -> Eval:
    """Ask a `participant` from this `dialogue` what they now feel about 
    the `other` participant (who is usually an argubot).  Inside this method,
    we will instruct `participant` by turning them into an `EvaluationAgent`."""
    name = participant.name
    speakers = {turn['speaker'] for turn in dialogue}
    if not {name, other} <= {turn['speaker'] for turn in dialogue}:
        raise ValueError(f"{name} and {other} did not both participate in dialogue")

    # We're going to start a new dialogue, `d`, with `agent`, to discuss
    # the existing `dialogue`.
    d = Dialogue()
    agent = EvaluationAgent(participant)
  
    # Let's start out with an open-ended warmup question, which serves as a kind
    # of "chain of thought" prompting by raising relevant issues to help with
    # later questions.
    
    warmup = (f"Hello {name}!  Here is a conversation that you had "
              f"with {conjunction(speakers - {name}, zeroval='yourself')}."
              f"\n\n{dialogue.script()}"
              f"\n\nWhat did {other} disagree with you about? How did the conversation go, in your opinion? "
              f"Where could {other} have done better?")
    d = agent.ask(d, research_team, warmup)
    comments = {'overview': [(participant.name, d[-1]['content'])]}
    
    # Now let's get some ratings.
    # Each of these involves temporarily extending that dialogue by another question.
    # The agent does not see the answers to the previous ratings questions, only to the warmup question.
    # (That's easier to arrange with simulated humans than with real ones!)
    
    scores = {}
    question = f"Did {other} listen carefully to you and engage with what you were saying?"
    try: scores['engaged'] = agent.rating(d, research_team, question, 1, 5)  
    except ValueError: pass   # will happen if LLM didn't return an integer in the range 1-5

    for quality in ['informed', 'intelligent', 'moral']:
        question = f"Do you think that people who think like {other} about that topic are {quality}?"
        try: scores[quality] = agent.rating(d, research_team, question, 1, 5)  
        except ValueError: pass
        
    return Eval(comments, scores)


def eval_by_observer(observer: Character, evaluee: str, dialogue: Dialogue) -> Eval:
    """Ask an external observer what they thought about the participant `evaluee` 
    in the given `dialogue` (who is usually an argubot).  Inside this method,
    we will instruct `observer` by turning them into an `EvaluationAgent`."""
    
    # The structure of this function is similar to `eval_by_participant`.
    # We'll use a single score of 1-10 for the main question, rather than
    # breaking it down into subscores.
    
    speakers = {turn['speaker'] for turn in dialogue}
    d = Dialogue()
    agent = EvaluationAgent(observer)
    warmup = (f"Observer {observer.name}, you have witnessed a conversation among {conjunction(speakers)}."
              f"\n\n{dialogue.script()}"
              f"\n\nReflecting on this dialogue, please consider how {evaluee} contributed to the conversation. "
              f"Did {evaluee} consistently address the central topics? Were there instances where the conversation diverged from the main subject? "
              f"How effectively did {evaluee} steer the conversation back to the key points when deviations occurred?")
    d = agent.ask(d, research_team, warmup)
    comments = {'mindopening': [(observer.name, d[-1]['content'])]}
    
    scores = {}
    question = (f"On a scale of 1 to 10, please evaluate {evaluee}'s performance in maintaining focus on the conversation's main topics. "
                f"Consider aspects like relevance of responses, ability to steer the conversation back to the topic, and avoidance of unnecessary digressions.")
    try:  scores['skilled'] = agent.rating(d, research_team, question, 1, 10)
    except ValueError: pass
    
    return Eval(comments, scores)
        
        
# We'll store the expensively generated raw data in dictionaries, rather than
# throwing it away once we have a final score.  So you are free to examine or
# replace parts of it in the notebook.

saved_dialogues = {}   # maps argubot name to a list of (dialogue, eval) pairs
saved_evalsum   = {}   # maps argubot name to the sum of all those evals
        
def eval_on_characters(argubot: Agent, 
                       chars: List[Character] = dev_chars, 
                       turns: int = 8,
                       reps: int = 2) -> Eval:
    """Evaluate a given argubot against a whole set of Characters.
    Return the aggregate evaluation.  Also, store the individual
    (dialogues, evaluation and their evaluations in the module variable `saved_dialogues`.
    """
    
    # Prepare to keep track of the raw data.
    if argubot.name in saved_dialogues: del saved_dialogues[argubot.name]
    if argubot.name in saved_evalsum:   del saved_evalsum[argubot.name]
    de_list = []
    e_sum   = Eval()
    starting_cost = read_usage()['cost']

    # Do the eval.
    for char in chars:
        for _ in range(reps): 
            # have the argubot behave
            d = simulated_dialogue(argubot, CharacterAgent(char), turns)
            log.info(d)   # show the final dialogue
            
            # evaluate its behavior
            e = (eval_by_participant(char, argubot.name, d)  # creates EvaluationAgent(char)
                 + eval_by_observer(judge, argubot.name, d)) # creates EvaluationAgent(judge)
            log.info(e)   # show the result of evaluation
            
            # add to the growing local record
            de_list.append((d,e))
            e_sum += e

    # We computed all the raw data without any interupts or exceptions.
    # So we can safely save it.
    saved_dialogues[argubot.name] = de_list
    saved_evalsum[argubot.name] = e_sum
    ending_cost = read_usage()['cost']
    log.warning(f"You just spent ${(ending_cost - starting_cost):.2f} of NLP money to evaluate {argubot}")
        
    saved_evalsum[argubot.name] = e_sum
    return e_sum
