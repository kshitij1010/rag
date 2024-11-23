import random
from dialogue import Dialogue
from agents import Agent

def simulated_dialogue(a: Agent, b: Agent, turns: int = 6, *,
                       prefix: Dialogue = Dialogue(),
                       starter=True) -> Dialogue:
    """Generate a simulated dialogue between Agents `a` and `b`, 
    for the given number of `turns`.  `a` goes first (following any supplied
    `prefix`).
    
    If `starter` is true, then `a` will try to use one of `b`'s conversation
    starters on the first turn, if any are defined. This is useful when `a` is
    an argubot and `b` is a `CharacterAgent`.
    """
    d = prefix
    if starter:
        # a tries to take a special first turn
        try:
            starters = b.conversation_starters  # type: ignore
            content = random.choice(starters)
            d = d.add(a.name, content)
            turns -= 1
            a, b = b, a   # switch roles
        except (AttributeError, TypeError, ValueError):
            pass
    
    while turns > 0:
        d = a.respond(d)
        turns -= 1
        a, b = b, a   # switch roles
    return d