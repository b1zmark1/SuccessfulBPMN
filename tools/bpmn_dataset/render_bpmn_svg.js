const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

function buildHtml() {
  const bundlePath = path.resolve(__dirname, 'node_modules', 'bpmn-js', 'dist', 'bpmn-navigated-viewer.development.js');
  const bundle = fs.readFileSync(bundlePath, 'utf-8');
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body, #canvas { width: 3000px; height: 3000px; margin: 0; padding: 0; }
  </style>
</head>
<body>
  <div id="canvas"></div>
  <script>${bundle}</script>
</body>
</html>`;
}

async function renderOne(page, xmlString) {
  return page.evaluate(async (xml) => {
    const ViewerCtor = window.BpmnJS && (window.BpmnJS.default || window.BpmnJS);
    if (typeof ViewerCtor !== 'function') {
      throw new Error(`BpmnJS is not a constructor (type=${typeof ViewerCtor})`);
    }
    const viewer = new ViewerCtor({ container: '#canvas' });
    try {
      await viewer.importXML(xml);
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      throw new Error(`Failed to import BPMN XML: ${msg}`);
    }
    const canvas = viewer.get('canvas');
    try {
      canvas.zoom(1.0);
    } catch (e) {
      // ignore
    }
    const viewbox = canvas.viewbox();
    const { svg } = await viewer.saveSVG();
    return { svg, viewbox };
  }, xmlString);
}

async function renderSingle(inputPath, outputSvgPath, outputMetaPath) {
  const xml = fs.readFileSync(inputPath, 'utf-8');
  const html = buildHtml();

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: 'load' });
    const result = await renderOne(page, xml);

    fs.mkdirSync(path.dirname(outputSvgPath), { recursive: true });
    fs.writeFileSync(outputSvgPath, result.svg, 'utf-8');
    fs.writeFileSync(outputMetaPath, JSON.stringify({ viewbox: result.viewbox }, null, 2), 'utf-8');
  } finally {
    await browser.close();
  }
}

async function renderBatch(tasksPath) {
  const tasks = JSON.parse(fs.readFileSync(tasksPath, 'utf-8'));
  const html = buildHtml();

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  try {
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: 'load' });

    for (const t of tasks) {
      const xml = fs.readFileSync(t.input, 'utf-8');
      const result = await renderOne(page, xml);
      fs.mkdirSync(path.dirname(t.outputSvg), { recursive: true });
      fs.writeFileSync(t.outputSvg, result.svg, 'utf-8');
      fs.writeFileSync(t.outputMeta, JSON.stringify({ viewbox: result.viewbox }, null, 2), 'utf-8');
    }
  } finally {
    await browser.close();
  }
}

async function main() {
  const args = process.argv.slice(2);
  if (args[0] === '--batch') {
    const tasksPath = args[1];
    if (!tasksPath) {
      console.error('Usage: node render_bpmn_svg.js --batch <tasks.json>');
      process.exit(1);
    }
    await renderBatch(tasksPath);
    return;
  }

  const input = args[0];
  const outSvg = args[1];
  const outMeta = args[2];
  if (!input || !outSvg || !outMeta) {
    console.error('Usage: node render_bpmn_svg.js <input.bpmn> <output.svg> <output_meta.json>');
    process.exit(1);
  }
  await renderSingle(input, outSvg, outMeta);
}

main().catch((err) => {
  console.error(err.stack || err.toString());
  process.exit(1);
});
