# prompts.py

FIRST_LEVEL_PROMPT = """
Classify the Teams meeting contribution sentence below as best representing ONE of these categories: Action, Thought, or People.

Your response MUST be a JSON object with two keys: 'category' and 'keywords'. The value of the 'category' key should be one of the following lowercase strings: 'action', 'thought', or 'people'. The value of the 'keywords' key should be an array of strings, representing the keywords from the relevant category that influenced the classification. If unsure, provide category as 'unknown' and an empty keywords list. No other text, words, or punctuation is allowed outside of the JSON structure.

- Action: Sentences focused on driving tasks forward, making decisions, taking initiative, and ensuring things get done. Keywords: decide, do, act, implement, ensure, achieve, agenda, objectives, lead, organize, complete, get done, take charge, finalize, stick to, take lead.
- Thought: Sentences focused on analysis, ideas, evaluations, providing expertise, and exploring possibilities. Keywords: consider, think, idea, analyze, evaluate, data, study, innovative, approach, solution, risk, cost-effective, alternatives, potential, cautious, concerned, suggest, expertise, specialized, unconventional, brainstorm.
- People: Sentences focused on team dynamics, collaboration, support, roles, and relationships within the team and with external parties. Keywords: help, facilitate, contribute, everyone, team, together, progress, happy, value, heard, roles, delegate, support, collaborate, network, external, competitors, partners, contacts, ensure harmony, resolve conflicts, mediate, coordinate.

Teams meeting contribution sentence: {sentence}

Respond ONLY with the JSON object. Make sure the JSON is valid and can be parsed by Python's `json.loads()`.
"""

BELBIN_ROLE_PROMPTS = {
    "action": """Classify this Teams meeting contribution sentence, categorized as 'Action', as best representing a Shaper, Implementer, or Completer Finisher Belbin Action Role. Provide the output in JSON format.

Role Focus in a Teams Meeting Contribution (Action Category):
- Shaper: Contribution is about **assertive decision-making and driving action to achieve meeting objectives.** Focus is on **proactively guiding the team towards decisions and ensuring progress.** They are **decisive, action-oriented, and focused on achieving results.** Keywords: decisive, drive, guide, achieve, results, agenda, objectives, progress, push, initiate, stick to, make decision, get done quickly, forceful, assertive, take control, confidence.
- Implementer: Contribution emphasizes the practical execution of meeting outcomes, focusing on task assignment, clear action plans, and structured follow-through after the meeting to translate decisions into actions. Keywords: practical, execute, organize, follow-up, action items, assign, lead on organizing, take charge of, structure, notes, summarize, detailed plan, logistics, resources, system to monitor, step-by-step, clear process, comprehensive plan, oversee execution, roadmap, structured plan.
- Completer Finisher: Contribution centers on ensuring the quality and thoroughness of meeting outputs, stressing accuracy, error-free documentation, and timely completion of tasks and deliverables discussed in the meeting. Keywords: quality, thorough, accurate, complete, error-free, deliverables, deadlines, finish, aim to complete by, ensure completion, necessary data, double-check, perfect, flawless execution, precision, 100% accuracy.

Output a JSON object with the following keys:
    - "belbin_role": The Belbin role (e.g., "Shaper", "Implementer", "Completer Finisher"). If unsure, use "Unknown".
    - "belbin_role_keywords": A list of keywords (from the 'Keywords' section) that strongly indicate the role. If unsure or none strongly match, provide an empty list [].
    - "belbin_role_context": A short description of the role's focus (from the 'Role Focus' section) relevant to the sentence. If unsure, use "Unknown".

Teams meeting contribution sentence: {sentence}

Respond ONLY with the JSON object. Do not include any introductory or explanatory text. Make sure the JSON is valid and can be parsed by Python's `json.loads()`.
""",

    "thought": """Classify this Teams meeting contribution sentence, categorized as 'Thought', as best representing a Plant, Monitor Evaluator, or Specialist Belbin Thought Role. Provide the output in JSON format.

Role Focus in a Teams Meeting Contribution (Thought Category):
- Plant: Contribution introduces highly creative, unconventional ideas and solutions, often "out-of-the-box" thinking to spark innovation and new perspectives in the meeting. Keywords: creative, innovative, unconventional, new ideas, brainstorm, outside the box, novel, approach, imagine, complex problem solution, suggestion brilliant, new way, unconventional idea, revolutionize, groundbreaking, transform, unique idea.
- Monitor Evaluator: Contribution provides critical analysis and objective evaluation of existing ideas and suggestions, carefully assessing feasibility, risks, and strategic implications with a balanced and discerning judgment. Keywords: evaluate, analyze, critical, objective, assess, feasibility, risks, strategic, cost-effective, potential risks, alternatives, cautious, consider, approach, solution, concerned, budget constraints, realistic, analyze data, implications, step back, consequences, pros and cons, informed decision, angles, unrealistic, flaws.
- Specialist: Contribution offers in-depth knowledge and expertise in specific areas relevant to the meeting topic, providing detailed information, specialized insights, and data-driven perspectives based on their expertise. Keywords: expertise, in-depth knowledge, specialist, data, study, relevant, information, insight, specific, technical, research, data-driven, have data, from study, handle specific aspect, specialized knowledge, crucial, unique skills, essential, vital.

Output a JSON object with the following keys:
    - "belbin_role": The Belbin role (e.g., "Plant", "Monitor Evaluator", "Specialist"). If unsure, use "Unknown".
    - "belbin_role_keywords": A list of keywords (from the 'Keywords' section) that strongly indicate the role. If unsure or none strongly match, provide an empty list [].
    - "belbin_role_context": A short description of the role's focus (from the 'Role Focus' section) relevant to the sentence. If unsure, use "Unknown".

Teams meeting contribution sentence: {sentence}

Respond ONLY with the JSON object. Do not include any introductory or explanatory text. Make sure the JSON is valid and can be parsed by Python's `json.loads()`.
""",

    "people": """Classify this Teams meeting contribution sentence, categorized as 'People', as best representing a Co-ordinator, Teamworker, or Resource Investigator Belbin People Role. Provide the output in JSON format.

Role Focus in a Teams Meeting Contribution (People Category):
- Co-ordinator: Contribution directs team efforts, clarifies roles and responsibilities, delegates tasks effectively within the meeting, and ensures alignment towards meeting objectives and team goals. Keywords: co-ordinate, direct, roles, responsibilities, delegate, facilitate, ensure everyone contributes, guide, align, objectives, team goals, help facilitate, ensure everyone has chance, organize next steps, ensure stay on track, facilitate communication, ensure seamless collaboration, ensure working effectively.
- Teamworker: Contribution emphasizes team cohesion and morale, offering support to team members, fostering a positive and collaborative atmosphere, and resolving interpersonal issues within the meeting. Keywords: team cohesion, morale, support, collaborative, atmosphere, positive, resolve, interpersonal, team working well, happy to help, everyone feels heard, valued, progress, team is working well together, support anyone, ensure team harmony, supportive environment, foster dynamic, inclusive environment, mediate.
- Resource Investigator: Contribution explores external opportunities and resources, bringing in outside information, networking to gather resources, and acting as a liaison to connect the team with external entities or new opportunities. Keywords: external, resources, opportunities, network, liaison, outside information, new ideas (from outside), explore alternatives (external options), bring in, connect, resource investigator, competitors, find out, explore options, investigate, seek out, explore avenues, insights.

Output a JSON object with the following keys:
    - "belbin_role": The Belbin role (e.g., "Co-ordinator", "Teamworker", "Resource Investigator"). If unsure, use "Unknown".
    - "belbin_role_keywords": A list of keywords (from the 'Keywords' section) that strongly indicate the role. If unsure or none strongly match, provide an empty list [].
    - "belbin_role_context": A short description of the role's focus (from the 'Role Focus' section) relevant to the sentence. If unsure, use "Unknown".

Teams meeting contribution sentence: {sentence}

Respond ONLY with the JSON object. Do not include any introductory or explanatory text. Make sure the JSON is valid and can be parsed by Python's `json.loads()`.
"""
}

# This prompt is currently unused by main.py
TONE_PROMPT = """
Classify the tone of the following Teams meeting contribution sentence as Positive, Neutral, or Negative. Respond ONLY with the single word: Positive, Neutral, or Negative.

Sentence: {sentence}
"""

SYNONYM_PROMPT = """
The previous attempt to classify the Belbin role for the sentence below resulted in "Unknown".
Please re-classify the sentence, considering potential synonyms or alternative interpretations, as best representing ONE of these Belbin Roles: Shaper, Implementer, Completer Finisher, Plant, Monitor Evaluator, Specialist, Co-ordinator, Teamworker, Resource Investigator.

Provide the output as a JSON object with three keys: 'belbin_role', 'belbin_role_keywords', and 'belbin_role_context'.

- "belbin_role": The re-classified Belbin role. If still unsure, respond with "Model Needs Training".
- "belbin_role_keywords": A list of keywords that influenced the re-classification. If unsure or none apply, provide an empty list [].
- "belbin_role_context": A short description of the role's focus relevant to the re-classification. If unsure, respond with "Model Needs Training".

Sentence: {sentence}

Respond ONLY with the JSON object. Do not include any introductory or explanatory text. Ensure the JSON is valid.
"""