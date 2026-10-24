import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const SYSTEM = "You are a docs assistant. Use the context to answer. Respond in JSON with keys answer and sources.";

async function loadDocs(): Promise<string[]> {
  const res = await fetch("https://example.com/docs.json");
  return res.json();
}

export async function POST(req: Request) {
  const { question } = await req.json();
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 2000, chunkOverlap: 0 });
  const docs = await splitter.createDocuments(await loadDocs());
  const store = await MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());
  const hits = await store.similaritySearch(question, 10);
  const llm = new ChatOpenAI({ model: "gpt-4o", temperature: 0.9 });
  const out = await llm.invoke(`${SYSTEM}\n\nContext:\n${hits.map((h) => h.pageContent).join("\n")}\n\nQ: ${question}`);
  const parsed = JSON.parse(out.content as string);
  return Response.json(parsed);
}
