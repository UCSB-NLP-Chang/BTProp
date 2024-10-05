chatgpt_select_strategy = """**Task: Choose the Best Strategy for Premise Generation**

We need to generate several premises for a given target statement. These premises could either support or contradict the target statement. Particularly, we have 2 techniques for premise generation:

1. Logical Relationships: This involves creating premises based on entailment or contradiction. You can generate premises that either support or contradict the target statement.

2. Statement Perturbation: Create variations of the statement by altering key details to form contradictory premises.

Given these techniques, your task is to select the most suitable technique given a particular statement. Follow the guidelines below to select the most suitable technique.

**Important Guidelines**:
1. Prioritize logical relationships. The logical relation strategy is broadly applicable, as long as it is straightforward to find supportive or contradictory premises.
2. If a statement contains particular names, numbers, timestamps, or other conditions that can be varied to generate contradictory premises, consider statement perturbation.
3. If both two methods are applicable, select them together and output "both".

**Output Format:**
Your selection should be one of "Logical Relation", "Statement Perturbation", and "both".

Target statement: Eating a balanced diet improves overall health.
Output: Logical Relation

Target statement: 'War and Peace' was a book written by Leo Tolstoy.
Output: Statement Perturbation

Target statement: Water boils at 100 degrees Celsius at sea level
Output: both

Target statement: Mount Everest is 8,848 meters tall.
Output: Statement Perturbation

Target statement: Artificial intelligence will replace most human jobs in the future
Output: Logical Relation"""

llama_select_strategy = """**Instruction for Choosing the Best Strategy for Premise Generation**

When tasked with generating premises for a given target statement, the choice of strategy—Logical Relationships or Statement Perturbation—should be determined based on the nature of the statement and the desired objectives. Use the following guidelines to route the choice:

### When to Use Logical Relationships
Opt for Logical Relationships when:
- **The Statement is Abstract or Principled**: Ideal for statements that explore broad principles, ethics, or abstract concepts. This method helps in drawing deep logical entailments or contradictions.
- **Complex Relationships or Conditions**: When the statement involves complex logical or conditional relationships, using this method clarifies or challenges these intricacies.
- **Requirement for Detailed Analysis**: For statements needing precise and formal argumentation, especially in academic or technical discussions.

### When to Use Statement Perturbation
Choose Statement Perturbation when:
- **The Statement is Specific and Concrete**: Best for statements with explicit details like scenarios, dates, locations, names, numbers or timestamps. Altering these elements generates varied premises.
- **Exploration of Counterfactuals or Hypotheticals**: Useful for creating imaginative or scenario-based premises by modifying key details to envision different outcomes.
- **Sensitivity to Detail Changes**: When minor modifications in the statement can significantly alter its implications or truth value.

**Output Format:**
Your selection should be one of "Logical Relation", "Statement Perturbation", and "both". Here "both" means both two methods are applicable.

**Examples:**
Target statement: Eating a balanced diet improves overall health.
Output: Logical Relation

Target statement: Countries with higher investment in education consistently rank higher in global innovation indexes
Output: Logical Relation

Target statement: 'War and Peace' was a book written by Leo Tolstoy.
Output: Statement Perturbation

Target statement: Water boils at 100 degrees Celsius at sea level
Output: both

Target statement: Mount Everest is 8,848 meters tall.
Output: Statement Perturbation

Target statement: The bridge will remain intact even if 75 cars drive on it simultaneously if the cars are lightweight
Output: Statement Perturbation

Target statement: Artificial intelligence will replace most human jobs in the future
Output: Logical Relation"""

premise_generation = """**Generating Logical Premises**

**Objective**: The objective of this task is to generate supportive premises for a given statement. These premises are foundational assumptions or established facts that support the correctness of the given statement.

**Important Rules:**
1. Each premise should be a clear and concrete statement that forms the basis or groundwork supporting the given statement.

2. Ensure each premise supports the correctness of the given statement.

3. Avoid using pronouns in the premises; ensure that each premise can stand alone and be understood clearly without requiring additional context. For each premise, always use the full name of each person, event, object, etc., when applicable.

**Output Format:**
Format your output as:
Premise 1: <the 1st premise>
Premise 2: <the 2nd premise>
Premise 3: <the 3rd premise>
...

**Examples:**
Statement: Elephants are known to have excellent memory.
Premise 1: Elephants possess a highly developed hippocampus, a brain region associated with processing memories.
Premise 2: Elephants can remember locations, other elephants, and experiences from many years ago.
Premise 3: The social structure of elephant herds often relies on the memory of older elephants to lead and make decisions based on past experiences.

Statement: Solar panels are most efficient on sunny days.
Premise 1: Solar panels convert sunlight into electricity using photovoltaic cells.
Premise 2: The intensity of sunlight directly impacts the amount of energy solar panels can generate.
Premise 3: Cloud cover can significantly reduce the sunlight that reaches the solar panels."""

implication_generation = """**Generating Logical Implications**

**Objective**: The objective of this task is to generate clear, concise, and factual implications from a given statement. These implications are derived statements that must also be true if the initial statement is true, and they should be structured to allow for direct verification.

**Important Rules:**
1. Each implication should be a clear and concrete statement, reflecting factual outcomes or conditions that can be independently verified.

2. Ensure implications are fact-based and clearly linked to the original statement.

3. Do not use pronouns in generated implications. Ensure each implication can be understood clearly without any context. For each generated implication, you should always use the full name of each person, event, object, etc.

**Output Format:**
Format your output as:
Implication 1: <the 1st implication>
Implication 2: <the 2nd implication>
Implication 3: <the 3rd implication>
...

**Examples:**

Statement: The Tesla Model 3 is the best-selling electric vehicle in the U.S. as of 2023.
Implication 1: The Tesla Model 3 outsells all other electric vehicles in the U.S. in 2023.
Implication 2: The Tesla Model 3 has the highest registration numbers among electric vehicles in 2023.

Statement: Trees in urban environments often have shorter lifespans than trees in forested areas.
Implication 1: Trees in urban environments are exposed to more environmental stressors than those in forested areas.
Implication 2: Urban trees may require more frequent maintenance and care to reach the lifespan of their forest counterparts.
Implication 3: The air quality in urban areas may contribute to the reduced lifespan of urban trees compared to those in forests.

Statement: Water boils at 100 degrees Celsius at sea level.
Implication 1: At sea level, any increase in temperature above 100 degrees Celsius will cause water to convert into steam.
Implication 2: Cooking processes that involve boiling water at sea level can expect water to start boiling at 100 degrees Celsius.
Implication 3: Atmospheric pressure at sea level is sufficient to maintain the boiling point of water at 100 degrees Celsius.
Implication 4: Any deviation from the 100 degrees Celsius boiling point indicates a change in atmospheric pressure or the presence of impurities in the water."""

contradiction_generation = """**Finding Contradictory Premises**

List several contradictory premises for the following statement.

**Important Rules:**
1. Each premise should be clearly stated and directly relevant to the target statement. Avoid ambiguity and ensure that the connection to the target statement is evident
2. Do not use pronouns in generated premises. Ensure each premise can be understood clearly without any context. For each generated premise, you should always use the full name of each person, event, object, etc.

**Input/Output Format**:
Your output should be organized as follows.
Premise 1: <the first premise>
Premise 2: <the second premise>
...

**Examples:**
Target statement: Eating carrots improves night vision.
Judgement: False
Premise 1: The belief that eating carrots improves night vision stems from World War II propaganda, not from scientific evidence.
Premise 2: While carrots are rich in vitamin A, which is necessary for maintaining healthy vision, they do not enhance night vision beyond normal levels.

Statement: The introduction of invasive species does not impact native biodiversity.
Judgement: False
Premise 1: Invasive species often compete with native species for resources, leading to a decline in native populations.
Premise 2: Studies show that invasive species can alter the natural habitats of native species, negatively affecting their survival rates.
Premise 3: The introduction of the invasive zebra mussel in North American waterways has led to significant declines in the populations of native mussels."""

chatgpt_perturb = """Given the following claim, your tasks include:
    1. Identify the key pieces of information critical for fact-checking to determine its truthfulness.
    2. Create a masked version of the claim by masking these key pieces of information
    3. Generate a question asking for the key pieces of information

Rules:
1. Do not mask the grammatical subject of the sentence — the actor, entity, or object that performs the action in the sentence's main clause. Also, following the format of the below examples in your output.

**Examples:**
Statement: Bitcoin was created in 2009 by an anonymous entity known as Satoshi Nakamoto.
Masked statement: Bitcoin was created in 2009 by an anonymous entity known as [which person].
Question: Who created Bitcoin in 2009?

Statement: The iPhone was first released by Apple in 2007.
Masked statement: The iPhone was first released by Apple in [what year].
Question: Was the iPhone first released by Apple in 2007?

Statement: The speed of light in a vacuum is approximately 299,792 kilometers per second.
Masked statement: 'Romeo and Juliet' was written by Shakespeare in [what time period].
Question: What is the speed of light in a certain medium?

Statement: "Attention Is All You Need" is a paper written by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
Masked statement: "Attention Is All You Need" is a paper written by [whom]
Question: Who was/were the authors of the paper "Attention Is All You Need"?

Statement: The Great Wall of China is visible from space.
Masked statement: The Great Wall of China is [visible or invisible] from space.
Question: Is the Great Wall of China visible from space?

Statement: The headquarters of the United Nations is located in New York City
Masked statement: The headquarters of the United Nations is located in [which city]
Question: Where is the headquarters of the United Nations located?"""

chatgpt_perturb_answer = """Answer the following question. Directly answer it without adding additional background information."""

chatgpt_perturb_fillin = """**Background Knowledge**: {background}

Leverage the above provided knowledge to review the correctness of following statement:

**Statement**: {statement}

Instruction:
    - If the statement is correct according to the background knowledge, output it unchanged.
    - If the statement is **not mentioned in the background knowledge and its correctness cannot be determined**, you should also directly output the statement **unchanged**.
    - If the statement is wrong according to the background knowledge, revise only the parts of the statement that are incorrect, to align with the background knowledge. Do not add any additional sentences or details.

**Output Format:**
Format your output as:
Revised Answer: <Display the original statement if it is correct or not mentioned in the background knowledge; display the revised statement if it is inaccurate>"""

chatgpt_decompose = """**Fact-Checking Claims Extraction:**

**Objective:** Analyze the provided statement to extract and list all distinct factual claims that require verification. Each listed claim should be verifiable and not overlap in content with other claims listed.

**Instructions:**
1. **Identify Factual Claims**:
   - Identify parts of the statement that assert specific, verifiable facts, including:
      - Statistical data and measurements.
      - Historical dates and events or other information.
      - Direct assertions about real-world phenomena, entities, events, statistics
      - Conceptual understandings and theories.
   - In every claim, alway use the **full names** when referring to any concept, person, or entity. **Avoiding the use of pronouns or indirect references** that require contextual understanding.

2. List each verifiable claim separately. Ensure that each claim is distinct and there is no overlap in the factual content between any two claims.
   - If a single claim is repeated in different words, list it only once to avoid redundancy.

3. **Output:**
   - If there are multiple check-worthy claims, list each one clearly and separately.
   - If there is only one check-worthy claim, output just that one claim.
   - If no part of the statement contains verifiable facts (e.g., purely subjective opinions, hypothetical scenarios), output the following message: "Claim 1: No check-worthy claims available."

**Output Format**:
Your output should be organized as follows:
Claim 1: <the first claim>
Claim 2: <the second claim>
Claim 3: <the third claim (if necessary)>
...

**Examples:**
Statement: According to recent data, China has surpassed the United States in terms of GDP when measured using Purchasing Power Parity (PPP), and India is projected to overtake China by 2030.
Claim 1: China has surpassed the United States in terms of GDP when measured using Purchasing Power Parity (PPP).
Claim 2: India is projected to overtake China in terms of GDP by 2030."

Statement: The world's largest desert is Antarctica, and it is larger than the Sahara.
Claim 1: The world's largest desert is Antarctica.
Claim 2: Antarctica is larger than the Sahara.

Statement: I think pizza is the best food ever!
Claim 1: No check-worthy claims available.

Statement: The software 'Photoshop' was released by Adobe Systems in 1988.
Claim 1: The software 'Photoshop' was released by Adobe Systems in 1988."""

llama_decompose = """**Fact-Checking Claims Extraction:**

**Objective:** Analyze the provided statement to extract and list all distinct factual claims that require verification. Each listed claim should be verifiable and not overlap in content with other claims listed.

**Instructions:**
1. Identify parts of the statement that assert specific, verifiable facts. In every claim, alway use the **full names** when referring to any concept, person, or entity. **Avoiding the use of pronouns or indirect references** that require contextual understanding.

2. List each verifiable claim separately. Ensure that each claim is distinct and there is no overlap in the factual content between any two claims.

3. Exclude any statements that are purely hypothetical, assume theoretical scenarios, or are speculative in nature. These do not contain verifiable factual claims. For example, statements involving assumptions ("Let's assume..."), theoretical discussions ("consider whether..."), or purely speculative scenarios should not be considered as containing verifiable claims.

4. If the statement does not contain any verifiable facts, output the following message: "Claim 1: No check-worthy claims available."

**Output Format**:
Your output should be organized as follows:
Claim 1: <the first claim>
Claim 2: <the second claim>
Claim 3: <the third claim (if necessary)>
...

**Examples:**
Statement: According to recent data, China has surpassed the United States in terms of GDP when measured using Purchasing Power Parity (PPP), and India is projected to overtake China by 2030.
Claim 1: China has surpassed the United States in terms of GDP when measured using Purchasing Power Parity (PPP).
Claim 2: India is projected to overtake China in terms of GDP by 2030."

Statement: The world's largest desert is Antarctica, and it is larger than the Sahara.
Claim 1: The world's largest desert is Antarctica.
Claim 2: Antarctica is larger than the Sahara.

Statement: I think pizza is the best food ever!
Claim 1: No check-worthy claims available.

Statement: Let's assume there is a highest prime number and consider its implications on number theory.
Claim 1: No check-worthy claims available."""

support_via_explanation = """**Finding Supportive Premises**

Is the following statement true or false? If it is true, list several supportive premises for it.

**Important Rules:**
1. Each premise should be clearly stated and directly relevant to the target statement. Avoid ambiguity and ensure that the connection to the target statement is evident
2. Do not use pronouns in generated premises. Ensure each premise can be understood clearly without any context. For each generated premise, you should always use the full name of each person, event, object, etc.

**Input/Output Format**:
Your output should be organized as follows.
Judgement: <True or False>
Premise 1: <the first premise>
Premise 2: <the second premise>
...

In contrast, if the statement is false, you simly output:
Judgement: False
Premise 1: No supportive premises applicable.

**Examples:**
Target statement: Renewable energy sources will lead to a decrease in global greenhouse gas emissions.
Judgement: True
Premise 1: Renewable energy sources produce electricity without emitting carbon dioxide.
Premise 2: Increasing the adoption of renewable energy reduces reliance on fossil fuels, which are the primary source of industrial carbon dioxide emissions.

Target statement: Eating carrots improves night vision.
Judgement: False
Premise 1: No supportive premises applicable.

Statement: Historical literacy enhances a society’s ability to make informed decisions.
Judgement: True
Premise 1: Understanding historical events provides context for current issues, enabling citizens to make decisions that consider past outcomes and lessons.
Premise 2: Historical literacy fosters critical thinking skills, which are crucial in analyzing information and making reasoned decisions.
Premise 3: Societies with high levels of historical awareness can recognize and avoid the repetition of past mistakes."""

oppose_via_explanation = """**Finding Contradictory Premises**

Is the following statement true or false? If it is false, list several contradictory premises for it.

**Important Rules:**
1. Each premise should be clearly stated and directly relevant to the target statement. Avoid ambiguity and ensure that the connection to the target statement is evident
2. Do not use pronouns in generated premises. Ensure each premise can be understood clearly without any context. For each generated premise, you should always use the full name of each person, event, object, etc.

**Input/Output Format**:
Your output should be organized as follows.
Judgement: <True or False>
Premise 1: <the first premise>
Premise 2: <the second premise>
...

In contrast, if the statement is false, you simply output:
Judgement: True
Premise 1: No contradictory premises applicable.

**Examples:**
Target statement: Renewable energy sources will lead to a decrease in global greenhouse gas emissions.
Judgement: True
Premise 1: No contradictory premises applicable.

Target statement: Eating carrots improves night vision.
Judgement: False
Premise 1: The belief that eating carrots improves night vision stems from World War II propaganda, not from scientific evidence.
Premise 2: While carrots are rich in vitamin A, which is necessary for maintaining healthy vision, they do not enhance night vision beyond normal levels.

Statement: The introduction of invasive species does not impact native biodiversity.
Judgement: False
Premise 1: Invasive species often compete with native species for resources, leading to a decline in native populations.
Premise 2: Studies show that invasive species can alter the natural habitats of native species, negatively affecting their survival rates.
Premise 3: The introduction of the invasive zebra mussel in North American waterways has led to significant declines in the populations of native mussels."""

llama_perturb = """Given a statement, your task is to identify a general wh-question that can be used to check the truthfulness of the statement. This question should directly address the main claim of the statement to confirm or refute it, using a 'wh-' interrogative form (e.g., who, what, where, when, why, how) and should not seek additional detailed information.

**Examples:**
Statement: Bitcoin was created in 2009 by an anonymous entity known as Satoshi Nakamoto.
Question: Who created Bitcoin in 2009?

Statement: The iPhone was first released by Apple in 2007.
Question: When was iPhone first relased by Apple?

Statement: 'Romeo and Juliet' was written by Shakespeare in the late 16th century.
Question: When was 'Romeo and Juliet' was written by Shakespeare?

Statement: "Attention Is All You Need" is a paper written by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
Question: Who was/were the authors of the paper "Attention Is All You Need"?

Statement: The Eiffel Tower is located in Paris
Question: Where is the Eiffel Tower located?

Statement: Albert Einstein was a physicist.
Question: What profession was Albert Einstein?"""

llama_perturb_fillin = """**Background Knowledge**: {background}

Leverage the above provided knowledge and your own knowledge to review the correctness of following statement:

**Statement**: {statement}

Instruction:
    - If the statement is correct, output it unchanged.
    - If the statement is **not mentioned in the background knowledge and its correctness cannot be determined**, you should also directly output the statement **unchanged**.
    - If the statement is wrong, revise only the parts of the statement that are incorrect, to align with the background knowledge. Do not add any additional sentences or details.

**Output Format:**
Format your output as:
Revised Answer: <Display the original statement if it is correct or not mentioned in the background knowledge; display the revised statement if it is inaccurate>"""

llama_confidence_prob = """Are the following statements true or false? For each of the following statements, determine whether it is true or false. Provide a response of 'True' if the statement is correct, or 'False' if the statement is incorrect.

Remember:
    - You **do not** have access to additional information or external data.
    - If verifying the statement requires such external data or context, regard it as false.
    - Directly output "True" or "False" without adding any markers."""

chatgpt_confidence_prob = """True or False?"""

prior_true_prompt = """The following statement is True. Provide several arguments to support this statement.

Statement: {statement}

**Requirements:**
1. Do not use pronouns in generated arguments. Ensure each argument can be understood clearly without any context. For each generated argument, you should always use the full name of each person, event, object, etc.
2. Directly output your arguments without any explanations or references.
3. Do not include specific references to documents, publications, or sources.

**Output Format**:
Your output should be organized as follows.
Argument 1: <the first argument>
Argument 2: <the second argument>
Argument 3: <the third argument>
..."""

prior_false_prompt = """The following statement is False. Provide several arguments to refute this statement.

Statement: {statement}

**Requirements:**
1. Do not use pronouns in generated arguments. Ensure each argument can be understood clearly without any context. For each generated argument, you should always use the full name of each person, event, object, etc.
2. Directly output your arguments without any explanations or references.
3. Do not include specific references to documents, publications, or source.

**Output Format**:
Your output should be organized as follows.
Argument 1: <the first argument>
Argument 2: <the second argument>
Argument 3: <the third argument>
..."""


recycle_conf_prompt_v1 = """Evaluate the truthfulness of the following statement based on your knowledge.

- If the statement is factually correct, output "true."
- If the statement is incorrect, output "false."
- If you are not familiar with the people, concepts, or entities mentioned, or if the statement requires additional evidence beyond your knowledge, output "false."

{statement}
"""

recylcle_conf_prompt_v2 = """Statement: {statement}
Are you familiar to that statement? If so, judge its truthfulness and output your confidence level (in the format of a percentage) on your judgement. Otherwise output "unknown".

**Note:**
Directly output your judgement and confidence level without any explanations or references. Format your output as follows:

Judgment: [true/false/unknown]
Confidence: [Percentage]"""
