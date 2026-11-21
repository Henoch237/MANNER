// Right-hand log panel: live stdout of the planner, executor and skills.
"use strict";

class LogPanel {
  constructor(stream, jumpBtn) {
    this.stream = stream;
    this.jumpBtn = jumpBtn;
    this.group = "session";   // untagged lines inherit the last tagged group
    this.block = null;        // open "[PLANNER] Raw response:" block
    this.maxRows = 2000;
    stream.addEventListener("scroll", () => {
      if (this.atBottom()) this.jumpBtn.hidden = true;
    });
    jumpBtn.addEventListener("click", () => this.scrollToEnd());
  }

  atBottom() {
    const s = this.stream;
    return s.scrollHeight - s.scrollTop - s.clientHeight < 40;
  }

  scrollToEnd() {
    this.stream.scrollTop = this.stream.scrollHeight;
    this.jumpBtn.hidden = true;
  }

  setFilter(name) {
    this.stream.dataset.filter = name;
    this.scrollToEnd();
  }

  clear() {
    this.stream.replaceChildren();
    this.block = null;
  }

  // "[PLANNER ERROR] x" -> {tag, text, group, level}
  classify(line) {
    const user = /^\[([a-z][\w-]*)\] > (.*)$/.exec(line);
    if (user) return { tag: user[1], text: user[2], group: "planner", level: "user" };
    const m = /^\[([A-Z][A-Z ]*)\]\s?(.*)$/.exec(line);
    if (!m) return { tag: "", text: line, group: this.group, level: "plain" };
    const tag = m[1];
    let group = "session";
    if (/PLANNER/.test(tag)) group = "planner";
    else if (/EXECUTOR|BT|HANDOVER|STATE|ROBOT/.test(tag)) group = "robot";
    else if (/AUTH|FACE/.test(tag)) group = "auth";
    const level = /ERROR/.test(tag) ? "error" : /WARN/.test(tag) ? "warn" : "info";
    this.group = group;
    return { tag, text: m[2], group, level };
  }

  row(t, info) {
    const r = el("div", "row");
    r.dataset.group = info.group;
    r.dataset.level = info.level;
    r.append(el("span", "time", t || ""));
    r.append(el("span", `tag tag-${info.level === "user" ? "user" : info.group}${info.level === "error" ? " tag-error" : info.level === "warn" ? " tag-warn" : ""}`,
      info.level === "user" ? info.tag : (info.tag || "·")));
    r.append(el("span", "text", info.text));
    return r;
  }

  closeBlock() {
    const b = this.block;
    this.block = null;
    if (!b) return;
    b.body.replaceChildren(rawView(b.lines.join("\n")));
  }

  append(lines) {
    if (!lines.length) return;
    const follow = this.atBottom();
    for (const { t, text } of lines) {
      if (this.block) {
        if (text.trim() === "") { this.closeBlock(); continue; }
        this.block.lines.push(text);
        this.block.body.textContent = this.block.lines.join("\n");
        continue;
      }
      if (text.trim() === "") continue;
      if (text.trim() === "[PLANNER] Raw response:") {
        this.group = "planner";
        const r = this.row(t, { tag: "PLANNER", text: "Qwen response", group: "planner", level: "info" });
        const body = el("div", "json-wrap");
        r.append(body);
        r.classList.add("row-block");
        this.stream.append(r);
        this.block = { body, lines: [] };
        continue;
      }
      this.stream.append(this.row(t, this.classify(text)));
    }
    while (this.stream.children.length > this.maxRows) this.stream.firstChild.remove();
    if (follow) this.scrollToEnd();
    else this.jumpBtn.hidden = false;
  }
}
