// End-to-end UI check with Playwright + the installed Microsoft Edge (no browser download).
//   node scripts/ui-check.mjs [baseUrl] [outDir] [--backend]
// Default: fixtures mode (no backend). --backend: analyze ../../cuda-archer through the real API.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const args = process.argv.slice(2);
const backend = args.includes("--backend");
const [base = "http://localhost:5173", out = "ui-shots"] = args.filter((a) => !a.startsWith("--"));
mkdirSync(out, { recursive: true });

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const shot = (name) => page.screenshot({ path: `${out}/${name}.png` });

try {
  await page.goto(backend ? base : `${base}/?fixtures=1`);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  check("home renders tagline", await page.getByText("See where it breaks.").isVisible());
  await shot("01-home");

  if (backend) {
    await page.getByText("Use a local folder instead").click();
    await page.getByRole("button", { name: /Analyze/ }).click();
  } else {
    await page.getByText("Try demo: CUDA Archer").click();
  }
  await page.getByText("Analyzing").waitFor({ timeout: 10_000 });
  await page.waitForTimeout(900);
  await shot("02-analyzing");

  await page.locator('.react-flow__node[data-id="retriever:retrieve"]').waitFor({ timeout: 60_000 });
  await page.waitForTimeout(3500); // reveal animation + fitView
  const nodeCount = await page.locator(".react-flow__node").count();
  check("graph renders the components", nodeCount >= 15, `${nodeCount}`);
  const hidden = await page.locator('.react-flow__node[style*="visibility: hidden"]').count();
  check("all components measured & visible", hidden === 0, `${hidden} hidden`);
  check("summary shows evaluation gaps", await page.getByText("evaluation gaps found").isVisible());
  await shot("03-workspace");

  await page.locator('.react-flow__node[data-id="retriever:retrieve"]').click();
  await page.getByText("WHY?").first().waitFor();
  const drawer = page.locator("aside").last();
  check("drawer shows WHY evidence DEFAULT_TOP_K = 10", await drawer.getByText("DEFAULT_TOP_K = 10").first().isVisible());
  check("drawer shows file:line", await drawer.getByText(/cuda_archer\/rag\/retriever\.py:\d+/).first().isVisible());
  await shot("04-why-drawer");
  await page.keyboard.press("Escape");

  await page.locator("button", { hasText: /Screen Input|Input Guardrail/ }).first().click();
  await page.waitForTimeout(800);
  check("hotspot chip selects node", (await page.locator('.react-flow__node.selected[data-id="guardrail:screen_input"]').count()) === 1);
  await page.keyboard.press("Escape");

  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await page.getByText("Apply recommended").waitFor();
  await page.waitForTimeout(700); // mode switch re-frames the graph
  await shot("05-eval-mode");

  const card = page.locator('[draggable="true"]', { hasText: "Context Precision" });
  await card.dragTo(page.locator('.react-flow__node[data-id="retriever:retrieve"]'));
  await page.waitForTimeout(300);
  const retrieverChips = page.locator('.react-flow__node[data-id="retriever:retrieve"]').getByText("Context Precision");
  check("drag & drop attaches evaluator", (await retrieverChips.count()) === 1);

  await card.dragTo(page.locator('.react-flow__node[data-id="guardrail:screen_input"]'));
  await page.waitForTimeout(300);
  check("invalid drop rejected with applies_to toast", await page.getByText(/applies to: retriever, reranker/).first().isVisible());
  check("invalid drop adds nothing", (await page.locator('.react-flow__node[data-id="guardrail:screen_input"]').getByText("Context Precision").count()) === 0);

  await page.getByRole("button", { name: /Apply recommended/ }).click();
  await page.getByText(/Added \d+ recommended evaluators|already attached/).first().waitFor({ timeout: 15_000 });
  await page.waitForTimeout(400);
  const chipCount = await page.locator(".react-flow__node button[aria-label^='Remove']").count();
  check("apply recommended adds chips", chipCount >= 15, `${chipCount} chips`);
  await shot("06-recommended");

  await page.locator(".react-flow__node button[aria-label^='Remove']").first().click();
  await page.waitForTimeout(300);
  const afterRemove = await page.locator(".react-flow__node button[aria-label^='Remove']").count();
  check("× removes a chip", afterRemove === chipCount - 1, `${chipCount} → ${afterRemove}`);

  await page.reload();
  await page.locator('.react-flow__node[data-id="retriever:retrieve"]').waitFor({ timeout: 30_000 });
  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await page.waitForTimeout(3000);
  const persisted = await page.locator(".react-flow__node button[aria-label^='Remove']").count();
  check("attachments persist across reload", persisted === afterRemove, `${persisted}`);

  await page.getByRole("button", { name: /Run Evaluation/ }).click();
  await page.waitForTimeout(700);
  await shot("07-run-progress");
  await page.getByText(/Run finished|Run failed/).first().waitFor({ timeout: backend ? 120_000 : 15_000 });
  check("run completes", await page.getByText(/Run finished/).first().isVisible());
  await page.waitForTimeout(1500); // badges pop in
  const badges = await page.locator(".react-flow__node .eg-pop[title*='n=']").count();
  check("result badges on the graph", badges >= 5, `${badges} badges`);
  const report = page.locator("aside", { hasText: "Run report" });
  check("run report opens", await report.isVisible());
  check("report lists bottlenecks", (await report.getByText(/Bottlenecks \(\d+\)/).count()) === 1);
  if (backend) {
    // results must say where they come from: MOCK (canned), REPLAY (recorded), or nothing for a live DIAL run
    const health = await (await page.request.get(`${base.replace(/:\d+$/, ":8000")}/api/health`).catch(() => null))?.json?.();
    const pill = await report.getByText(/^(REPLAY · MOCK|REPLAY|MOCK)$/).first().textContent().catch(() => null);
    const expected = health?.replay ? /REPLAY/ : health?.mock ? /MOCK/ : null;
    check("provenance label next to results", expected ? expected.test(pill ?? "") : pill === null, pill ?? "live");
  }
  await shot("08-run-results");

  await page.locator('.react-flow__node[data-id="retriever:retrieve"]').click();
  const results = page.locator("aside", { hasText: /Results ·/ });
  await results.waitFor({ timeout: 5000 });
  check("drawer shows node results", await results.getByText("Context Precision").first().isVisible());
  if (backend) {
    await results.locator("li button").first().click();
    check("sample drill-down shows the question", await results.getByText(/Question · /).first().isVisible());
  }
  await shot("09-node-results");
  await page.locator("aside button[aria-label='Close']").click();

  if (backend) {
    await page.getByRole("button", { name: /Report/ }).click().catch(() => {});
    const fix = page.getByRole("button", { name: /Try suggested fix/ }).first();
    await fix.waitFor({ timeout: 5000 });
    await fix.click();
    await page.getByText(/Fix run finished|Run failed/).first().waitFor({ timeout: 120_000 });
    await page.waitForTimeout(800);
    check("fix run opens before/after", await page.getByText(/unchanged or not comparable/).isVisible());
    await shot("10-compare");
    await page.locator("button[aria-label='Close compare']").click();
    await page.getByRole("button", { name: "Before", exact: true }).click();
    check("before/after toggle", await page.getByRole("button", { name: "After", exact: true }).isVisible());
  }
  check("no uncaught page errors", errors.length === 0, errors.join(" | "));
} catch (e) {
  check("script completed", false, e.message.split("\n")[0]);
  await shot("99-failure");
} finally {
  await browser.close();
}

const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
