import type { ConfigValue, Unit } from "../types/contracts";

export function formatConfig(value: ConfigValue): string {
  if (value === null) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "string") return `"${value}"`;
  return String(value);
}

export function formatValue(value: number | null, unit: Unit): string {
  if (value === null) return "n/a";
  switch (unit) {
    case "score":
      return `${Math.round(value * 100)}%`;
    case "ms":
      return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`;
    case "usd":
      return value === 0 || value >= 0.01 ? `$${value.toFixed(3)}` : `$${value.toFixed(4)}`;
    case "tokens":
      return `${Math.round(value)} tok`;
  }
}

export function shortCommit(commit: string | null): string {
  return commit ? commit.slice(0, 7) : "";
}

export function fileLine(file: string | null, line: number | null): string {
  if (!file) return "";
  return line ? `${file}:${line}` : file;
}
