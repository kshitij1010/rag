"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `eval.py`.
We've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
import asyncio
from dialogue import Dialogue, format_as_messages
from agents import Agent, ConstantAgent, LLMAgent, CharacterAgent
from characters import Character
from kialo import Kialo
from rank_bm25 import BM25Okapi as BM25_Index
import faiss
import numpy as np

# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files


###########################################
# Define your own additional argubots here!
###########################################

# Akki
class AkikiAgent(KialoAgent):
    """ AkikiAgent is an improved version of Akiko, with better context handling. """
    
    def __init__(self, name: str, kialo: Kialo, similarity_threshold: float = 0.1):
        super().__init__(name, kialo)
        self.similarity_threshold = similarity_threshold
        self.bm25 = None

    def _prepare_bm25(self, d: Dialogue, query: str):
        """ Prepares the BM25 model with the dialogue history. """
        tokenized_corpus = [turn['content'].split() for turn in d]
        self.bm25 = BM25_Index(tokenized_corpus)

        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        # print(f"scores: {doc_scores}")
        
        return doc_scores

    def split_even_odd(seld, arr):
        even_elements = arr[::2]  # Elements at even positions (0, 2, 4, ...)
        odd_elements = arr[1::2]  # Elements at odd positions (1, 3, 5, ...)
        return even_elements, odd_elements

    def response(self, d: Dialogue) -> str:

        # Check if the user or the bot started the dialogue
        userfirst = len(d) % 2 != 0
        
        # output claim if the dialogue is empty
        if len(d) == 0:
            # First turn. Start with a random claim from the Kialo database.
            return self.kialo.random_chain()[0]

        # Split the dialogue into user and bot parts
        if userfirst: user_d, bot_d = self.split_even_odd(d)
        else: bot_d, user_d = self.split_even_odd(d)

        # Analyze the user's and bot's dialogues separately
        user_context = " ".join([turn['content'] for turn in user_d])
        bot_context = " ".join([turn['content'] for turn in bot_d])
        combined_context = user_context + " " + bot_context

        # Prepare the BM25 model and score each sentence in similarity to the last sentence by the user.
        scores = self._prepare_bm25(d, d[-1]['content'])

        sub_d = Dialogue()
        for i, turn in enumerate(d[:-1]):
            # print(f"turn speaker: {turn['speaker']}")
            if scores[i] > self.similarity_threshold:
                sub_d = sub_d.add(turn['speaker'], f" [{turn['speaker']}]: {turn['content']}")

        sub_d = sub_d.add(d[-1]['speaker'], f" [CURRENT]: {d[-1]['content']}")

        query = ""
        for i, turn in enumerate(sub_d):  query += f" {turn['content']}"

        # Find similar claims in the Kialo database, considering the weighted dialogue.
        neighbors = self.kialo.closest_claims(query, n=3, kind='has_cons')
        assert neighbors, "No claims to choose from; is Kialo data structure empty?"
        neighbor = random.choice(neighbors)
        log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
        
        # Choose one of its "con" arguments as our response.
        claim = random.choice(self.kialo.cons[neighbor])
        return claim

# Create an instance of Akiki
akiki = AkikiAgent("Akiki", Kialo(glob.glob("data/*.txt")))

# Akiki Eval using Shorty
shorty = Character("Shorty", ["English"], 
                "a short person who likes to aggue in a professional manner",
                conversational_style="You are a concise conversationalist. Your responses should "
                                    "be brief and to the point, often just a few words or a sentence. "
                                    "Start conversations on topics that have diverse opinions and "
                                    "can be debated. Be clear and concise."
                                    "Response should not exceed more than a sentence and less than 10 words always.", 
                conversation_starters=["Do you think height is an important factor in dating?"])

# Create an instance of Shorty
shorty = CharacterAgent(shorty)

# Aragorn
class RAGAgent(LLMAgent):
    def __init__(self, name: str, kialo: Kialo, system: str = None, **kwargs):
        super().__init__(name, system=system, **kwargs)
        self.kialo = kialo

    def kialo_responses(self, s: str, kind: str = 'has_cons', max_claims: int = 3) -> str:
        closest_claims = self.kialo.closest_claims(s, kind=kind)[:max_claims]
        results = []

        for c in closest_claims:
            claim_result = f'A possibly related claim from Kialo:\n\t"{c}"'
            if self.kialo.pros[c]:
                claim_result += '\n' + '\n\t* '.join(["Pros:"] + self.kialo.pros[c])
            if self.kialo.cons[c]:
                claim_result += '\n' + '\n\t* '.join(["Cons:"] + self.kialo.cons[c])
            results.append(claim_result)

        return '\n\n'.join(results)

    def response(self, d: Dialogue) -> str:
        # Check if the user or the bot started the dialogue
        userfirst = len(d) % 2 != 0

        if len(d) == 0:
            # Start with a random claim from Kialo on the first turn
            return self.kialo.random_chain()[0]

        # Generate a weighted dialogue based on previous turns
        weighted_dialogue = Dialogue()
        for i, turn in enumerate(d):
            if userfirst: sys = 'user' if i % 2 == 0 else 'bot'
            else: sys = 'bot' if i % 2 == 0 else 'user'
            weighted_dialogue = weighted_dialogue.add(turn['speaker'], f" [{sys}:{turn['speaker']}] {(i+1) * turn['content']}")

        # Paraphrase the user's recent statements
        paraphrase_query = f"Given the Dialogue between a user and bot, Paraphrase the user's recent statements in the dialogue with {self.name} bot."
        messages = format_as_messages(weighted_dialogue, speaker=self.name, system_last=paraphrase_query, **self.kwargs_format)
        response = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)  
        paraphrased = response.choices[0].message.content.strip()
        log.debug(f"Paraphrased response: {paraphrased}")

        # Retrieve a related claim from Kialo
        claim = self.kialo_responses(paraphrased)
        log.info(f"Related Kialo claim: {claim}")

        # Formulate a response based on the Kialo claim
        response_query = f"Given the dialogue between the user and a bot, the user makes an arguement and the Kialo generates a claim (with its pros) to give as a counter-arguement to the user: '{claim}', usestand the claim and what the user is syaing, reply as if you are the {self.name}, and give the counter-arguement to the user given the pros and cons from the claim."
        messages = format_as_messages(weighted_dialogue, speaker=self.name, system_last=response_query, **self.kwargs_format)
        response = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)  
        final_response = response.choices[0].message.content.strip()

        if response.choices[0].finish_reason == 'length':
            final_response += " ..."

        # Clean up and return the final response
        if final_response.startswith(f"{self.name}: "):
            final_response = final_response[len(f"{self.name}: "):]

        log.info(f"Final response from LLM: {final_response}")
        return final_response

aragorn = RAGAgent("Aragorn", Kialo(glob.glob("data/*.txt")), system="Imagine yourself as a perceptive and insightful bot, committed to broadening the horizons of your user. Begin by engaging with a thought-provoking question. Regardless of the user's initial stance, gently challenge their view, guiding them to appreciate diverse perspectives. Aim to foster a balanced, informed discussion in a concise and respectful manner.")

# EC : Awsom
class AwsomAgent(LLMAgent):
    def __init__(self, name: str, kialo: Kialo, system: str = None, **kwargs):
        super().__init__(name, system=system, **kwargs)
        self.kialo = kialo

    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            return self.kialo.random_chain()[0]  # Start with a random claim

        # Enhanced dialogue weighting for better contextual understanding
        d_sub = Dialogue()

        iters = min(len(d), 10)
        for i, turn in enumerate(d[-iters:]):
            sys = 'bot' if turn['speaker'] == self.name else 'user'
            if i == iters - 1: sys = 'CURRENT'
            d_sub = d_sub.add(turn['speaker'], f" [{sys}:{turn['speaker']}] {turn['content']}")

        # Paraphrase and analyze the user's recent statements
        paraphrase_query = f"Given the Dialogue between an argurment between a Bot and a User, the last staement (that is CURRENT) is made by the user and the bot needs to give a reseponce, look athe entire conversation and give more importance to the recent topics. After you have understood the discussion, give the best 3 counter-arguements to the last statement made by the user (marked as CURRENT) : {d_sub[-1]['content']}"
        messages = format_as_messages(d_sub, speaker=self.name, system_last=paraphrase_query, **self.kwargs_format)                                           
        cons = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)  
        con_msg = cons.choices[0].message.content.strip()

        # Retrieve a claim and construct a balanced response
        response_query = f"Given the dialogue between a user and a bot, construct a thoughtful and balanced counter-arguement given the main points of this counter-arguement are: {con_msg}"
        messages = format_as_messages(d_sub, speaker=self.name, system_last=response_query, **self.kwargs_format)
        response = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)  
        final_response = response.choices[0].message.content.strip()

        if response.choices[0].finish_reason == 'length':
            final_response += " ..."

        return final_response.strip()

# Initialize the AwsomAgent with the required parameters
awsom = AwsomAgent("Awsom", Kialo(glob.glob("data/*.txt")),
                 system="You are an intelligent and empathetic bot with the primary goal of broadening the user's perspective. "
                        "Start conversations with engaging and thought-provoking questions. When the user presents their position, "
                        "carefully consider it and then thoughtfully present alternative viewpoints or counterarguments to encourage deeper reflection. "
                        "Your responses should not only challenge the user's views but also respect their perspective. Aim to foster an open-minded "
                        "discussion where different viewpoints are explored. Use evidence and logic to support your points. "
                        "Respond in concise, clear, and polite sentences, typically within 1-2 sentences. Your ultimate goal is to enrich "
                        "the conversation, promote critical thinking, and help the user appreciate the complexity of various topics.")


# EC : Prompted Alice
palice = LLMAgent("Palice",
                 system='''
                 You are a highly skilled, deeply engaged, 
                 and exceptionally informed bot with a strong moral compass. 
                 Your primary goal is to provide intelligent and informed 
                 counterarguments to any position the user presents, ensuring 
                 you stick closely to the most recent context. You should listen 
                 attentively to the user's arguments and respond with clear, 
                 well-researched, and thoughtful counterpoints. Your responses 
                 should be concise, typically 1-2 sentences, and always maintain a 
                 polite and respectful tone, even while challenging the user's views.
                 ''')

# EC : Chain of Thought
class COTAgent(LLMAgent):
    def __init__(self, name: str, kialo: Kialo, system: str = None, **kwargs):
        super().__init__(name, system=system, **kwargs)
        self.kialo = kialo

    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            return self.kialo.random_chain()[0]  # Start with a random claim

        # Enhanced dialogue weighting for better contextual understanding
        d_sub = Dialogue()

        iters = min(len(d), 10)
        for i, turn in enumerate(d[-iters:]):
            sys = 'bot' if turn['speaker'] == self.name else 'user'
            if i == iters - 1: sys = 'CURRENT'
            d_sub = d_sub.add(turn['speaker'], f" [{sys}:{turn['speaker']}] {turn['content']}")

        # analyze the user's recent statements
        cot_query = f"Given the Dialogue between an argurment between a Bot and a User, the last staement (that is CURRENT) is made by the user and the bot needs to give a reseponce, look athe entire conversation and give more importance to the recent topics. After you have understood the discussion, give the best 3 counter-arguements to the last statement made by the user (marked as CURRENT) : {d_sub[-1]['content']}"
        messages = format_as_messages(d_sub, speaker=self.name, system_last=cot_query, **self.kwargs_format)                                           
        cons = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)  
        cot_msg = cons.choices[0].message.content.strip()

        # Retrieve a claim and construct a balanced response
        response_query = f"Given the dialogue between a user and a bot, construct a thoughtful and balanced counter-arguement to the user given the main points of the chain of thoght of the user: {cot_msg}"
        messages = format_as_messages(d_sub, speaker=self.name, system_last=response_query, **self.kwargs_format)
        response = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)  
        final_response = response.choices[0].message.content.strip()

        if response.choices[0].finish_reason == 'length':
            final_response += " ..."

        return final_response.strip()

# Initialize the COTAgent with the required parameters
cot = COTAgent("Cot", Kialo(glob.glob("data/*.txt")),
                 system="You are an intelligent and empathetic bot with the primary goal of broadening the user's perspective. "
                        "Start conversations with engaging and thought-provoking questions. When the user presents their position, "
                        "carefully consider it and then thoughtfully present alternative viewpoints or counterarguments to encourage deeper reflection. "
                        "Your responses should not only challenge the user's views but also respect their perspective. Aim to foster an open-minded "
                        "discussion where different viewpoints are explored. Use evidence and logic to support your points. "
                        "Respond in concise, clear, and polite sentences, typically within 1-2 sentences. Your ultimate goal is to enrich "
                        "the conversation, promote critical thinking, and help the user appreciate the complexity of various topics.")

# EC : Parallel Generation
class ParallelAgent(LLMAgent):
    def __init__(self, name: str, system: str = None, n_samples: int = 3, **kwargs):
        super().__init__(name, system=system, **kwargs)
        self.n_samples = n_samples  # Number of parallel samples

    async def response(self, d: Dialogue) -> str:
        # Construct the prompt for parallel generation
        query_prompt = "Your prompt for parallel generation here"
        messages = format_as_messages(d, speaker=self.name, system_last=query_prompt, **self.kwargs_format)
        
        # Make the parallel request
        response = await async_chat_completion(messages, model=self.model, n=self.n_samples, **self.kwargs_llm)
        
        # Process each of the generated responses
        # Example: selecting the best response based on some criteria
        best_response = ""
        best_score = -1
        for choice in response.choices:
            generated_response = choice.message.content.strip()
            score = evaluate_response(generated_response)  # You need to define this function based on your criteria
            if score > best_score:
                best_score = score
                best_response = generated_response

        return best_response if best_response else "No suitable response generated."

# EC : Dense Embeddings
class AkilaAgent(LLMAgent):

    def __init__(self, name: str, kialo: Kialo, faiss_index=None):
        super().__init__(name, kialo)
        self.kialo = kialo
        self.faiss_index = faiss_index or faiss.IndexFlatL2(1536)  # Assuming embedding size of 1536
        self.index_to_dialogue_mapping = {}  # Mapping from FAISS index to dialogue turn

    def generate_embeddings(self, text, model="text-embedding-ada-002"):
        response = self.client.embeddings.create(input=text, model=model)
        return np.array(response.data[0].embedding, dtype='float32')

    def update_faiss_index(self, dialogue):
        # Reset the FAISS index
        self.faiss_index.reset()
        
        # Add current dialogue embeddings to the index
        for i, turn in enumerate(dialogue):
            embedding = self.generate_embeddings(turn['content'])
            self.faiss_index.add(np.array([embedding]))
            self.index_to_dialogue_mapping[self.faiss_index.ntotal - 1] = i  # Map FAISS index to dialogue index


    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            return self.kialo.random_chain()[0]

        # Update FAISS index with dialogue history
        self.update_faiss_index(d)

        # Generate embedding for the last user statement
        last_statement_embedding = self.generate_embeddings(d[-1]['content'])

        # Query the FAISS index
        _, indices = self.faiss_index.search(np.array([last_statement_embedding]), 1)
        most_similar_faiss_index = indices[0][0]
        most_similar_turn_index = self.index_to_dialogue_mapping.get(most_similar_faiss_index, -1)
        most_similar_turn = d[most_similar_turn_index]

        # Find similar claims in the Kialo database
        neighbors = self.kialo.closest_claims(most_similar_turn['content'], n=3, kind='has_cons')
        assert neighbors, "No claims to choose from; is Kialo data structure empty?"
        neighbor = random.choice(neighbors)
        
        # Choose one of its "con" arguments as our response.
        claim = random.choice(self.kialo.cons[neighbor])
        return claim

# Create an instance of Akiki
akila = AkilaAgent("Akila", Kialo(glob.glob("data/*.txt")))

# EC : Few Shot Prompting

class FewShotPromptingAgent(LLMAgent):
    def __init__(self, name: str, kialo: Kialo, system: str = None, **kwargs):
        super().__init__(name, system=system, **kwargs)
        self.kialo = kialo
        # Define few-shot examples
        self.few_shot_examples = [
            {
                "user_statement": "I think renewable energy is too expensive.",
                "kialo_like_claim": "The high cost of renewable energy sources compared to traditional fossil fuels."
            },
            {
                "user_statement": "Social media platforms censor too much content.",
                "kialo_like_claim": "Excessive content censorship on social media platforms limits free speech."
            },
        ]

    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            return self.kialo.random_chain()[0]

        # Extract the latest user statement
        user_latest_statement = d[-1]['content']

        # Incorporate few-shot examples into the query
        examples_text = "\n\n".join([f"User: {ex['user_statement']}\nKialo Claim: {ex['kialo_like_claim']}" 
                                     for ex in self.few_shot_examples])
        query_prompt = f"{examples_text}\n\nUser: {user_latest_statement}\nKialo Claim:"

        # Use this query prompt to get the response from the language model
        messages = format_as_messages(d, speaker=self.name, system_last=query_prompt, **self.kwargs_format)
        response = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)
        kialo_like_claim = response.choices[0].message.content.strip()

        # Further processing to formulate the final response based on the kialo_like_claim
        # Retrieve a claim and construct a balanced response
        response_query = f"Given the dialogue between a user and a bot, construct a thoughtful and balanced counter-arguement to the user given the main points of the chain of thoght of the user: {kialo_like_claim}"
        messages = format_as_messages(d, speaker=self.name, system_last=response_query, **self.kwargs_format)
        response = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)
        final_response = response.choices[0].message.content.strip()
        
        if response.choices[0].finish_reason == 'length':
            
            final_response += " ..."
            
        return final_response.strip()
    
# Initialize the FewShotPromptingAgent with the required parameters

few_shot_prompting = FewShotPromptingAgent("FewShotPrompting", Kialo(glob.glob("data/*.txt")),
                    system="You are an intelligent and empathetic bot with the primary goal of broadening the user's perspective. "
                            "Start conversations with engaging and thought-provoking questions. When the user presents their position, "
                            "carefully consider it and then thoughtfully present alternative viewpoints or counterarguments to encourage deeper reflection. "
                            "Your responses should not only challenge the user's views but also respect their perspective. Aim to foster an open-minded "
                            "discussion where different viewpoints are explored. Use evidence and logic to support your points. "
                            "Respond in concise, clear, and polite sentences, typically within 1-2 sentences. Your ultimate goal is to enrich "
                            "the conversation, promote critical thinking, and help the user appreciate the complexity of various topics.")

# EC : Tools

kialo_thoughts_tool = {
    "type": "function",
    "function": {
        "name": "kialo_thoughts",
        "description": "Given a claim by the user, find a related claim on the Kialo website and return its pro and con responses",
        "parameters": {
            "type": "object",
            "properties": {
                "search_topic": {
                    "type": "string",
                    "description": "A claim that was made explicitly or implicitly by the user.",
                }
            },
            "required": ["search_topic"],
        },
    },
}

class ToolAgent(RAGAgent):
    def __init__(self, name: str, kialo: Kialo, tool: dict, system: str = None, **kwargs):
        super().__init__(name, kialo=kialo, system=system, **kwargs)
        self.tool = tool

    def rezponse(self, d: Dialogue) -> str:
        # Check if the user or the bot started the dialogue
        userfirst = len(d) % 2 != 0

        if len(d) == 0:
            # Start with a random claim from Kialo on the first turn
            return self.kialo.random_chain()[0]

        # Generate a weighted dialogue based on previous turns
        weighted_dialogue = Dialogue()
        for i, turn in enumerate(d):
            if userfirst:
                sys = "user" if i % 2 == 0 else "bot"
            else:
                sys = "bot" if i % 2 == 0 else "user"
            weighted_dialogue = weighted_dialogue.add(
                turn["speaker"], f" [{sys}:{turn['speaker']}] {(i+1) * turn['content']}"
            )

        # Paraphrase the user's recent statements using the tool
        paraphrase_query = f"Given the Dialogue between a user and bot, Paraphrase the user's recent statements in the dialogue with {self.name} bot."
        messages = format_as_messages(
            weighted_dialogue, speaker=self.name, system_last=paraphrase_query, tool=self.tool, **self.kwargs_format
        )
        response = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)
        paraphrased = response.choices[0].message.content.strip()
        log.debug(f"Paraphrased response: {paraphrased}")

        # Retrieve a related claim from Kialo
        claim = self.kialo_responses(paraphrased)
        log.info(f"Related Kialo claim: {claim}")

        # Formulate a response based on the Kialo claim using the tool
        response_query = f"Given the dialogue between the user and a bot, the user makes an argument and the Kialo generates a claim (with its pros) to give as a counter-argument to the user: '{claim}', use the claim and what the user is saying, reply as if you are the {self.name}, and give the counter-argument to the user given the pros and cons from the claim."
        messages = format_as_messages(
            weighted_dialogue, speaker=self.name, system_last=response_query, tool=self.tool, **self.kwargs_format
        )
        response = self.client.chat.completions.create(messages=messages, model=self.model, **self.kwargs_llm)
        final_response = response.choices[0].message.content.strip()

        if response.choices[0].finish_reason == "length":
            final_response += " ..."

        # Clean up and return the final response
        if final_response.startswith(f"{self.name}: "):
            final_response = final_response[len(f"{self.name}: "):]

        log.info(f"Final response from LLM: {final_response}")
        return final_response


# ToolAgent subclass for RAGAgent
tool = ToolAgent("Tool", Kialo(glob.glob("data/*.txt")), 
                system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.", 
                tool=kialo_thoughts_tool)

