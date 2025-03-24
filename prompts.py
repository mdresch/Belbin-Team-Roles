# prompts.py

FIRST_LEVEL_PROMPT = """
Classify the Teams meeting contribution sentence below as best representing ONE of these categories: Action, Thought, or People.

Your response MUST be a JSON object with two keys: 'category' and 'keywords'. The value of the 'category' key should be one of the following lowercase strings: 'action', 'thought', or 'people'. The value of the 'keywords' key should be an array of strings, representing the keywords from the relevant category that influenced the classification. No other text, words, or punctuation is allowed outside of the JSON structure.

- Action: Sentences focused on driving tasks forward, making decisions, taking initiative, and ensuring things get done. Keywords: decide, do, act, implement, ensure, achieve, agenda, objectives, lead, organize, complete.
- Thought: Sentences focused on analysis, ideas, evaluations, providing expertise, and exploring possibilities. Keywords: consider, think, idea, analyze, evaluate, data, study, innovative, approach, solution, risk, cost-effective, alternatives, potential.
- People: Sentences focused on team dynamics, collaboration, support, roles, and relationships within the team and with external parties. Keywords: help, facilitate, contribute, everyone, team, together, progress, happy, value, heard, roles, delegate, support, collaborate, network, external.

Teams meeting contribution sentence: [SENTENCE_HERE]
"""

BELBIN_ROLE_PROMPTS = {
    "action": """Classify this Teams meeting contribution sentence, categorized as 'Action', as best representing a Shaper, Implementer, or Completer Finisher Belbin Action Role. Provide the output in JSON format.

Role Focus in a Teams Meeting Contribution (Action Category):
- Shaper: Contribution is about **assertive decision-making and driving action to achieve meeting objectives.** Focus is on **proactively guiding the team towards decisions and ensuring progress.** They are **decisive, action-oriented, and focused on achieving results.** Keywords: decisive, drive, guide, achieve, results, agenda, objectives, progress, push, initiate.
- Implementer: Contribution emphasizes the practical execution of meeting outcomes, focusing on task assignment, clear action plans, and structured follow-through after the meeting to translate decisions into actions. Keywords: practical, execute, organize, follow-up, action items, assign, lead, take charge of, structure.
- Completer Finisher: Contribution centers on ensuring the quality and thoroughness of meeting outputs, stressing accuracy, error-free documentation, and timely completion of tasks and deliverables discussed in the meeting. Keywords: quality, thorough, accurate, complete, error-free, deliverables, deadlines, finish, aim to complete by, ensure completion.

Output a JSON object with the following keys:
    - "belbin_role": The Belbin role (e.g., "Shaper", "Implementer", "Completer Finisher").
    - "belbin_role_keywords": A list of keywords associated with the role (from the 'Keywords' section).
    - "belbin_role_context": A short description of the role's focus (from the 'Role Focus' section).

    Example Output:
    {
        "belbin_role": "Shaper",
        "belbin_role_keywords": ["decisive", "drive", "guide", "achieve", "results", "agenda", "objectives", "progress", "push", "initiate"],
        "belbin_role_context": "Contribution proactively guides the team towards decisions and ensures progress, focusing on achieving results."
    }

    Example of a sentence that is NOT a Shaper:
    "Let's make sure we document all action items carefully."
""",

    "thought": """Classify this Teams meeting contribution sentence, categorized as 'Thought', as best representing a Plant, Monitor Evaluator, or Specialist Belbin Thought Role. Provide the output in JSON format.

Role Focus in a Teams Meeting Contribution (Thought Category):
- Plant: Contribution introduces highly creative, unconventional ideas and solutions, often "out-of-the-box" thinking to spark innovation and new perspectives in the meeting. Keywords: creative, innovative, unconventional, new ideas, brainstorm, outside the box, novel, approach, imagine.
- Monitor Evaluator: Contribution provides critical analysis and objective evaluation of existing ideas and suggestions, carefully assessing feasibility, risks, and strategic implications with a balanced and discerning judgment. Keywords: evaluate, analyze, critical, objective, assess, feasibility, risks, strategic, cost-effective, potential risks, alternatives, cautious, consider, approach, solution.
- Specialist: Contribution offers in-depth knowledge and expertise in specific areas relevant to the meeting topic, providing detailed information, specialized insights, and data-driven perspectives based on their expertise. Keywords: expertise, in-depth knowledge, specialist, data, study, relevant, information, insight, specific, technical, research, data-driven, have data, from study.

Output a JSON object with the following keys:
    - "belbin_role": The Belbin role (e.g., "Plant", "Monitor Evaluator", "Specialist").
    - "belbin_role_keywords": A list of keywords associated with the role.
    - "belbin_role_context": A short description of the role's focus.

    Example Output:
    {
        "belbin_role": "Monitor Evaluator",
        "belbin_role_keywords": ["evaluate", "analyze", "critical", "objective", "assess", "feasibility", "risks", "strategic", "cost-effective", "potential risks", "alternatives", "cautious", "consider", "approach", "solution"],
        "belbin_role_context": "Contribution provides critical analysis and objective evaluation of existing ideas and suggestions, carefully assessing feasibility, risks, and strategic implications with a balanced and discerning judgment."
    }
""",

    "people": """Classify this Teams meeting contribution sentence, categorized as 'People', as best representing a Co-ordinator, Teamworker, or Resource Investigator Belbin People Role. Provide the output in JSON format.

Role Focus in a Teams Meeting Contribution (People Category):
- Co-ordinator: Contribution directs team efforts, clarifies roles and responsibilities, delegates tasks effectively within the meeting, and ensures alignment towards meeting objectives and team goals. Keywords: co-ordinate, direct, roles, responsibilities, delegate, facilitate, ensure everyone contributes, guide, align, objectives, team goals, help facilitate.
- Teamworker: Contribution emphasizes team cohesion and morale, offering support to team members, fostering a positive and collaborative atmosphere, and resolving interpersonal issues within the meeting. Keywords: team cohesion, morale, support, collaborative, atmosphere, positive, resolve, interpersonal, team working well, happy to help, everyone feels heard, valued, progress, team is working well together.
- Resource Investigator: Contribution explores external opportunities and resources, bringing in outside information, networking to gather resources, and acting as a liaison to connect the team with external entities or new opportunities. Keywords: external, resources, opportunities, network, liaison, outside information, new ideas (from outside), explore alternatives (external options), bring in, connect, resource investigator.

Output a JSON object with the following keys:
    - "belbin_role": The Belbin role (e.g., "Co-ordinator", "Teamworker", "Resource Investigator").
    - "belbin_role_keywords": A list of keywords associated with the role.
    - "belbin_role_context": A short description of the role's focus.

    Example Output:
    {
        "belbin_role": "Teamworker",
        "belbin_role_keywords": ["team cohesion", "morale", "support", "collaborative", "atmosphere", "positive", "resolve", "interpersonal", "team working well", "happy to help", "everyone feels heard", "valued", "progress", "team is working well together"],
        "belbin_role_context": "Contribution emphasizes team cohesion and morale, offering support to team members, fostering a positive and collaborative atmosphere, and resolving interpersonal issues within the meeting."
    }
"""
}


TONE_PROMPTS = {
    "Teamworker": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
    Sentence: [SENTENCE_HERE]
""",
    "Implementer": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
    Sentence: [SENTENCE_HERE]
""",
    "Shaper": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
    Sentence: [SENTENCE_HERE]
""",
    "Co-ordinator": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
    Sentence: [SENTENCE_HERE]
""",
    "Resource Investigator": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
     Sentence: [SENTENCE_HERE]
""",
    "Plant": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
    Sentence: [SENTENCE_HERE]
""",
    "Monitor Evaluator": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
    Sentence: [SENTENCE_HERE]
""",
    "Specialist": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
    Sentence: [SENTENCE_HERE]
""",
    "Completer Finisher": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
    Sentence: [SENTENCE_HERE]
""",
     "Unknown": """Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Return ONLY the single word: Positive, Neutral, or Negative.
     Sentence: [SENTENCE_HERE]
"""
}