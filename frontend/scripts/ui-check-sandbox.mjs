// Browser check of the generic-repo path (no adapter): Run setup → Docker sandbox run → rule-based patch → compare.
//   node scripts/ui-check-sandbox.mjs [baseUrl] [outDir]     (backend + Docker running; mock mode is fine)
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const [base = "http://localhost:5173", out = "ui-shots-sandbox"] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const shot = (name) => page.screenshot({ path: `${out}/${name}.png` });

try {
  await page.goto(base);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByText("Use a local folder instead").click();
  await page.locator("input").first().fill("tests/runtime/fixtures/rag_app");
  await page.getByRole("button", { name: /Analyze/ }).click();
  await page.locator('.react-flow__node[data-id="retriever:retrieve"]').waitFor({ timeout: 90_000 });
  await page.waitForTimeout(2500);
  check("hand-rolled retriever on the graph", true);

  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await page.getByRole("button", { name: /Run setup/ }).click();
  const setup = page.locator("aside", { hasText: "Run setup" });
  await setup.waitFor();
  check("run setup shows the inferred function", (await setup.locator("input").nth(1).inputValue()) === "answer");
  const before = await setup.locator("textarea").count();
  await setup.getByText("add", { exact: true }).click();
  await setup.locator("textarea").last().fill("How long does the boiler take to heat up?");
  await shot("01-run-setup");
  await setup.getByRole("button", { name: /Save run setup/ }).click();
  await page.getByText(/Run setup saved/).first().waitFor({ timeout: 10_000 });
  check("run setup saves an added question", true, `${before - 1} → ${before} textareas`);

  await page.getByRole("button", { name: /Apply recommended/ }).click();
  await page.getByText(/Added \d+ recommended evaluators|already attached/).first().waitFor({ timeout: 15_000 });
  await page.getByRole("checkbox").uncheck().catch(() => {}); // full (small) generated set
  await page.getByRole("button", { name: /Run Evaluation/ }).click();
  await page.getByText(/Run finished|Run failed/).first().waitFor({ timeout: 300_000 });
  check("sandbox run completes", await page.getByText(/Run finished/).first().isVisible());
  await page.waitForTimeout(1200);
  await shot("02-sandbox-results");

  await page.locator('.react-flow__node[data-id="retriever:retrieve"]').click();
  const drawer = page.locator("aside", { hasText: "Suggested patch" });
  await drawer.waitFor({ timeout: 10_000 });
  check("hotspot shows a rule-based patch", await drawer.getByText(/TOP_K 10 → 4/).first().isVisible());
  check("patch is downloadable", (await drawer.locator("a[download]").count()) > 0);
  await shot("03-patch");
  await drawer.getByRole("button", { name: /Try this fix/ }).first().click();
  await page.getByText(/Fix run finished|Run failed/).first().waitFor({ timeout: 300_000 });
  await page.waitForTimeout(800);
  check("patched run opens before/after", await page.getByText(/unchanged or not comparable/).isVisible());
  check("compare header names the patch", await page.locator("span.font-mono", { hasText: "as-is + 1 patch" }).first().isVisible());
  await shot("04-compare");

  // sweep the same literal over its suggested values, against the as-is run
  await page.locator("button[aria-label='Close compare']").click();
  await page.getByRole("button", { name: "Before", exact: true }).click();
  await page.locator('.react-flow__node[data-id="retriever:retrieve"]').click();
  const sweepBtn = page.getByRole("button", { name: /Sweep TOP_K/ }).first();
  await sweepBtn.waitFor({ timeout: 10_000 });
  await sweepBtn.click();
  await page.getByText(/Sweep finished|Sweep failed/).first().waitFor({ timeout: 600_000 });
  check("sweep finishes", await page.getByText(/Sweep finished/).first().isVisible());
  const header = page.getByText(/Sweep of/).first();
  check("sweep table opens", await header.isVisible());
  const cols = await page.locator("th", { hasText: "TOP_K =" }).count();
  check("one column per value + the code's value", cols === 5, `${cols} columns`);
  await shot("05-sweep");
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
