import { useEffect } from "react";
import { Toasts } from "./components/Toasts";
import { Analyzing } from "./screens/Analyzing";
import { Home } from "./screens/Home";
import { Workspace } from "./screens/Workspace";
import { useAppStore } from "./store/useAppStore";

export default function App() {
  const { screen, analysis, loadHealth, openAnalysis, select } = useAppStore();

  useEffect(() => {
    void loadHealth();
    const match = /^#\/a\/([\w]+)$/.exec(window.location.hash);
    if (match) void openAnalysis(match[1]);
  }, [loadHealth, openAnalysis]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") select(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [select]);

  return (
    <>
      {screen === "home" && <Home />}
      {screen === "analyzing" && <Analyzing />}
      {screen === "workspace" && analysis && <Workspace analysis={analysis} />}
      <Toasts />
    </>
  );
}
