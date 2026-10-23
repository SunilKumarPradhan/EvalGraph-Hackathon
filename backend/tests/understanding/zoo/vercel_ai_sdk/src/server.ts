import { openai } from "@ai-sdk/openai";
import { generateText, tool } from "ai";
import express from "express";
import { z } from "zod";

const app = express();
app.use(express.json());

const weather = tool({
  description: "Get the weather for a city",
  parameters: z.object({ city: z.string() }),
  execute: async ({ city }) => (await fetch(`https://wttr.in/${city}?format=3`)).text(),
});

async function answer(prompt: string) {
  const { text } = await generateText({
    model: openai("gpt-4o"),
    temperature: 1,
    tools: { weather },
    maxSteps: 5,
    prompt,
  });
  return text;
}

app.post("/chat", async (req, res) => {
  res.json({ answer: await answer(req.body.prompt) });
});

app.listen(3000);
