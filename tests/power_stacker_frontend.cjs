const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../js/lora-power-stacker.js"), "utf8")
    .replace(/^import .*;\r?\n/gm, "");

async function setup({ delayed = false } = {}) {
    const menus = [];
    const requests = [];
    let extension, resolveRequest, now = 0;
    const response = { ok: true, json: async () => ({ LoraLoader: {
        input: { required: { lora_name: [["A.safetensors", "B.safetensors"]] } },
    } }) };
    const pending = new Promise(resolve => { resolveRequest = resolve; });
    const fetchList = async url => {
        requests.push(url);
        return delayed ? pending : response;
    };
    const app = {
        registerExtension(value) { extension = value; },
        graph: { setDirtyCanvas() {} }, canvas: { ds: { scale: 1 } },
    };
    const context = vm.createContext({ app, api: { fetchApi: fetchList }, fetch: fetchList,
        console, Date: { now: () => now }, prompt: () => "0.3",
        LGraphNode: class {}, LiteGraph: { ContextMenu: class {
            constructor(items, options) { menus.push({ items, ...options }); }
        } },
    });
    vm.runInContext(source, context);
    class Node {
        constructor() { this.widgets = []; this.size = [420, 240]; this.pos = [100, 200]; this.registered = 0; }
        addWidget(type, name, value, callback) { this.widgets.push({ type, name, value, callback }); }
        addCustomWidget(widget) {
            this.registered++;
            // ComfyUI adopts custom widgets into LegacyWidget, preserving their methods.
            for (const [name, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(Object.getPrototypeOf(widget)))) {
                if (name !== "constructor") Object.defineProperty(widget, name, descriptor);
            }
            Object.setPrototypeOf(widget, Object.prototype);
            this.widgets.push(widget);
            return widget;
        }
        computeSize() { return [420, 24 * this.widgets.length]; }
        setSize(value) { this.size = value; }
    }
    await extension.beforeRegisterNodeDef(Node, { name: "PM LoRA Power Stacker" }, app);
    const node = new Node();
    node.onNodeCreated();
    const rows = [node.addLoraWidget(), node.addLoraWidget(), node.addLoraWidget()];
    const ctx = { fillRect() {}, strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
        fillText() {}, measureText: value => ({ width: value.length * 6 }) };
    rows.forEach((row, index) => row.draw(ctx, node, 420, 80 + index * 24, 20));
    return { node, rows, menus, requests, resolve: () => resolveRequest(response), advance: ms => { now += ms; } };
}

const tests = {
    async "independent row selections and clearing"() {
        const { node, rows, menus } = await setup();
        for (let i = 0; i < rows.length; i++) {
            await rows[i].openLoraSelector(node, {});
            menus.at(-1).callback({ content: `${i}.safetensors` });
        }
        assert.deepEqual(rows.map(row => row.value.lora), ["0.safetensors", "1.safetensors", "2.safetensors"]);
        await rows[1].openLoraSelector(node, {});
        menus.at(-1).callback("None");
        assert.deepEqual(rows.map(row => row.value.lora), ["0.safetensors", null, "2.safetensors"]);
    },
    async "targeted request shared by rows and refreshed after expiry"() {
        const h = await setup();
        await h.rows[0].openLoraSelector(h.node, {});
        await h.rows[1].openLoraSelector(h.node, {});
        assert.deepEqual(h.requests, ["/object_info/LoraLoader"]);
        h.advance(60001);
        await h.rows[2].openLoraSelector(h.node, {});
        assert.equal(h.requests.length, 2);
    },
    async "only the latest pending row opens a picker"() {
        const h = await setup({ delayed: true });
        const first = h.rows[0].openLoraSelector(h.node, {});
        const second = h.rows[1].openLoraSelector(h.node, {});
        h.resolve();
        await Promise.all([first, second]);
        assert.equal(h.requests.length, 1);
        assert.equal(h.menus.length, 1);
        h.menus[0].callback("B.safetensors");
        assert.deepEqual(h.rows.map(row => row.value.lora), [null, "B.safetensors", null]);
    },
    async "stale and removed-row menu callbacks cannot overwrite another selection"() {
        const h = await setup();
        await h.rows[0].openLoraSelector(h.node, {});
        const oldMenu = h.menus.at(-1);
        await h.rows[1].openLoraSelector(h.node, {});
        oldMenu.callback("A.safetensors");
        assert.equal(h.rows[0].value.lora, null);
        const removedMenu = h.menus.at(-1);
        h.node.removeLoraWidget(h.rows[1]);
        removedMenu.callback("B.safetensors");
        assert.equal(h.rows[1].value.lora, null);
    },
    async "drawn row bounds do not overlap and controls affect only their row"() {
        const h = await setup();
        const event = { type: "pointerdown", button: 0 };
        assert.equal(h.rows[0].mouse(event, [60, h.rows[1].lastY + 1], h.node), false);
        assert.equal(h.rows[1].mouse(event, [15, h.rows[1].lastY + 3], h.node), true);
        assert.deepEqual(h.rows.map(row => row.value.on), [true, false, true]);
        h.rows[2].adjustStrength("strength", -0.05);
        assert.deepEqual(h.rows.map(row => row.value.strength), [1, 1, 0.95]);
    },
    async "registered widgets survive workflow serialization and restoration"() {
        const h = await setup();
        assert.equal(h.node.registered, 3);
        assert.equal(new Set(h.rows.map(row => row.name)).size, 3);
        h.rows[1].value = { on: false, lora: "B.safetensors", strength: 0.3 };
        const workflow = {};
        h.node.onSerialize(workflow);
        assert.equal(workflow.widgets_values.length, 4);
        h.node.onConfigure(JSON.parse(JSON.stringify(workflow)));
        const restored = h.node.widgets.filter(widget => widget.type === "pm_lora_widget");
        assert.equal(restored.length, 3);
        assert.deepEqual(JSON.parse(JSON.stringify(restored[1].value)), { on: false, lora: "B.safetensors", strength: 0.3 });
    },
};

(async () => {
    let failed = 0;
    for (const [name, test] of Object.entries(tests)) {
        try { await test(); console.log(`PASS ${name}`); }
        catch (error) { failed++; console.error(`FAIL ${name}: ${error.message}`); }
    }
    process.exitCode = failed ? 1 : 0;
})();
