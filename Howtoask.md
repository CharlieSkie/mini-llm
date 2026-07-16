# How to Ask Questions About TMC

This mini model can answer better when the training data is focused and the questions are close to the topic it learned.

In this project, `data/train.txt` is currently focused on Trinidad Municipal College (TMC). That means the model is expected to generate answers related to TMC when you ask about:

- Trinidad Municipal College
- TMC history
- the establishment of the college
- people involved in the creation of the college
- important years and events
- courses and recognition mentioned in the training text

## Important Reminder

This project uses a small character-level language model. It does not understand meaning as deeply as a large chatbot model.

It learns patterns from text. If the training data is small, incomplete, or inconsistent, the model may:

- mix facts
- produce incomplete answers
- generate wrong dates or names
- copy only parts of the training text
- answer better for familiar wording than unfamiliar wording

## Can It Answer Different Wording With the Same Meaning?

Yes, it can sometimes answer even if the question uses different words, as long as the point is close to what it learned.

For example, these questions are similar:

```text
What is TMC?
Tell me about Trinidad Municipal College.
What do you know about TMC?
Can you explain the history of TMC?
```

These all point to the same topic, so the model has a better chance of answering about Trinidad Municipal College.

However, it is not guaranteed. To improve the model, add many question-and-answer examples with different wording.

## Best Way to Ask

Use clear and direct questions.

Good examples:

```text
What is Trinidad Municipal College?
When did the plan to establish a college in Trinidad start?
Who initiated the plan to establish a college in Trinidad?
What was Trinidad Institute of Technology?
When was Trinidad Junior College registered?
When did Trinidad Junior College become Trinidad Municipal College?
Who was the mayor when TMC started to grow?
How many students did TMC have in academic year 2022-2023?
What courses does TMC offer?
```

Avoid very short or unclear prompts.

Less effective examples:

```text
TMC
history
college
who
when
tell me
```

Short prompts can work sometimes, but the model may generate random or incomplete text.

## Recommended Training Format

For better answers, format `data/train.txt` like this:

```text
Topic: Trinidad Municipal College
Question: What is TMC?
Answer: TMC stands for Trinidad Municipal College, a higher education institution in Trinidad, Bohol.

Topic: TMC History
Question: When did the plan to establish a college in Trinidad start?
Answer: The plan started in 1980 when Mr. Paciano Petarco initiated the project of establishing a college institution in Trinidad.

Topic: TMC Founder
Question: Who initiated the plan to establish a college in Trinidad?
Answer: Mr. Paciano Petarco initiated the plan in 1980 and sought help from the Municipal Mayor, Atty. Avelino N. Puracan.
```

## Add Paraphrased Questions

To help the model answer different wording, add multiple versions of the same question.

Example:

```text
Topic: TMC History
Question: When did the vision for TMC begin?
Answer: The vision began in 1980 when Mr. Paciano Petarco initiated the plan to establish a college in Trinidad.

Question: When did the plan to establish a college in Trinidad start?
Answer: The plan started in 1980 when Mr. Paciano Petarco initiated the project.

Question: Who started the idea of creating a college in Trinidad?
Answer: Mr. Paciano Petarco started the idea in 1980.
```

The answer can be similar, but the questions should use different wording.

## Tips for Better TMC Answers

- Keep the training data focused on TMC.
- Use consistent `Topic`, `Question`, and `Answer` labels.
- Add many examples with different question wording.
- Keep answers factual and not too long.
- Clean encoding errors such as strange characters before training.
- Retrain the model after changing `data/train.txt`.

## After Updating train.txt

After editing `data/train.txt`, train the model again:

```bash
python train.py
```

Then test it with:

```bash
python agent.py
```
