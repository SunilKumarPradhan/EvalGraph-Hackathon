import {
  ArrowUpDown, Binary, Box, Braces, Database, FileText, Globe, History, LogIn, Search, ShieldCheck, Sparkles, Split,
  Table, Workflow, Wrench, type LucideIcon,
} from "lucide-react";
import type { NodeType, Severity } from "../../types/contracts";

export const TYPE_STYLE: Record<NodeType, { icon: LucideIcon; color: string; name: string }> = {
  entrypoint: { icon: LogIn, color: "#60a5fa", name: "Entrypoint" },
  agent: { icon: Workflow, color: "#a78bfa", name: "Agent" },
  router: { icon: Split, color: "#f472b6", name: "Router" },
  llm: { icon: Sparkles, color: "#2dd4bf", name: "LLM" },
  prompt: { icon: FileText, color: "#94a3b8", name: "Prompt" },
  retriever: { icon: Search, color: "#fbbf24", name: "Retriever" },
  vector_db: { icon: Database, color: "#f59e0b", name: "Vector DB" },
  embedding: { icon: Binary, color: "#fcd34d", name: "Embedding" },
  reranker: { icon: ArrowUpDown, color: "#fb923c", name: "Reranker" },
  tool: { icon: Wrench, color: "#4ade80", name: "Tool" },
  memory: { icon: History, color: "#38bdf8", name: "Memory" },
  guardrail: { icon: ShieldCheck, color: "#f87171", name: "Guardrail" },
  external_api: { icon: Globe, color: "#22d3ee", name: "External API" },
  parser: { icon: Braces, color: "#c084fc", name: "Parser" },
  dataset: { icon: Table, color: "#94a3b8", name: "Dataset" },
  other: { icon: Box, color: "#94a3b8", name: "Other" },
};

export const SEVERITY_COLOR: Record<Severity, string> = { high: "#ef4444", medium: "#f59e0b", low: "#eab308" };
export const SEVERITY_RANK: Record<Severity, number> = { high: 0, medium: 1, low: 2 };
export const SEVERITY_EMOJI: Record<Severity, string> = { high: "🔴", medium: "🟠", low: "🟡" };
