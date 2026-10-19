"""Physics engine menu.

Opens on the list of formulas the engine uses, with a "Launch engine" button that starts physics_engine.py as its own
program (a separate Python process, the same call lab_gl.bat makes).  Nothing of the engine runs in this window.

The formulas are read out of physics_engine.py's text (its FORMULAS table) without importing or running it, so this menu
opens instantly and always shows what the engine file currently contains.

Put this file in the same folder as physics_engine.py and run:  python physics_menu.py   (or double-click menu.bat)
"""
import ast
import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "physics_engine.py")
ENGINE_ARGS = ["--lab", "--gl", "--no-embed"]          # exactly what lab_gl.bat passes

BG, SIDE, PANEL, CARD, LINE = "#0d1015", "#11151b", "#151a21", "#1b212a", "#272e39"
FG, DIM, ACCENT, SEL, GOOD = "#e8ebf0", "#8b95a3", "#5b9cff", "#223552", "#4ade80"


def load_formulas(path):
    """The engine's FORMULAS table: ((section, ((name, equation, where it lives), ...)), ...).  Read from the file's text."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FORMULAS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("no FORMULAS table found in " + os.path.basename(path))


class Menu:
    def __init__(self, root):
        self.root = root
        root.title("Physics engine")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1100, int(sw * 0.8)), min(760, int(sh * 0.85))
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")
        root.minsize(720, 480)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        try:
            ttk.Style(root).theme_use("clam")
        except tk.TclError:
            pass
        try:
            self.formulas, self.problem = load_formulas(ENGINE), ""
        except (OSError, SyntaxError, ValueError) as e:
            self.formulas, self.problem = (), f"Could not read the formulas from physics_engine.py:\n{e}"
        self.status = tk.StringVar(value="")
        self._build()

    def _build(self):
        r = self.root
        n = sum(len(items) for _, items in self.formulas)
        side = tk.Frame(r, bg=SIDE, width=270)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        side.grid_rowconfigure(1, weight=1)
        side.grid_columnconfigure(0, weight=1)
        tk.Label(side, text="SECTIONS", bg=SIDE, fg=ACCENT, font=("Segoe UI", 9, "bold"), anchor="w",
                 padx=14).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(14, 6))
        self.lb = tk.Listbox(side, bg=SIDE, fg=FG, selectbackground=SEL, selectforeground="#ffffff", activestyle="none",
                             relief="flat", bd=0, highlightthickness=0, font=("Segoe UI", 10), exportselection=False)
        lsb = ttk.Scrollbar(side, command=self.lb.yview)
        self.lb.configure(yscrollcommand=lsb.set)
        self.lb.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 10))
        lsb.grid(row=1, column=1, sticky="ns", pady=(0, 10))
        self.lb.bind("<<ListboxSelect>>", self.jump)

        body = tk.Frame(r, bg=PANEL, padx=16, pady=12)
        body.grid(row=0, column=1, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(2, weight=1)
        tk.Label(body, text="Formulas", bg=PANEL, fg=FG, font=("Segoe UI", 14, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        tk.Label(body, text=f"Every formula the engine uses ({n}), with the function that holds it.",
                 bg=PANEL, fg=DIM, font=("Segoe UI", 10), anchor="w").grid(row=1, column=0, sticky="w", pady=(2, 0))
        right = tk.Frame(body, bg=PANEL)
        right.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="ne")
        self.launch_b = tk.Button(right, text="Launch engine", command=self.launch, bg=ACCENT, fg="#08111f",
                                  activebackground="#7bb0ff", activeforeground="#08111f", relief="flat", bd=0,
                                  font=("Segoe UI", 11, "bold"), width=14, pady=8, cursor="hand2", highlightthickness=0)
        self.launch_b.grid(row=0, column=0, sticky="e")
        # Build mode was taken out of the engine: the tab stays here, beside Launch engine, greyed out, and does nothing
        tk.Button(right, text="Build mode", state="disabled", bg=CARD, fg=DIM, disabledforeground=DIM, relief="flat", bd=0,
                  font=("Segoe UI", 11, "bold"), width=14, pady=8, highlightthickness=0).grid(row=0, column=1, sticky="e", padx=(8, 0))
        tk.Label(right, text="not in use", bg=PANEL, fg=DIM, font=("Segoe UI", 9, "italic"),
                 anchor="center").grid(row=1, column=1, sticky="ew", pady=(3, 0))
        tk.Label(right, textvariable=self.status, bg=PANEL, fg=DIM, font=("Segoe UI", 9),
                 anchor="e").grid(row=2, column=0, columnspan=2, sticky="e", pady=(3, 0))

        t = self.txt = tk.Text(body, wrap="word", bg=SIDE, fg=FG, relief="flat", font=("Consolas", 10), padx=14, pady=10,
                               highlightthickness=0, cursor="arrow", spacing1=1)
        sb = ttk.Scrollbar(body, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        sb.grid(row=2, column=1, sticky="ns", pady=(8, 0))
        t.tag_configure("sec", foreground=ACCENT, font=("Segoe UI", 12, "bold"), spacing1=14, spacing3=4)
        t.tag_configure("name", foreground=FG, font=("Segoe UI", 10, "bold"), spacing1=8)
        t.tag_configure("eq", foreground=GOOD, lmargin1=18, lmargin2=18)
        t.tag_configure("where", foreground=DIM, font=("Segoe UI", 9), lmargin1=18, lmargin2=18)
        t.tag_configure("bad", foreground="#f87171")
        if self.problem:
            t.insert("end", self.problem + "\n", "bad")
        for i, (sec, items) in enumerate(self.formulas):
            t.mark_set(f"sec{i}", "end-1c")
            t.mark_gravity(f"sec{i}", "left")                   # the mark stays at the start of the heading
            t.insert("end", sec + "\n", "sec")
            self.lb.insert("end", f"{sec}   ({len(items)})")
            for name, eq, where in items:
                t.insert("end", name + "\n", "name")
                t.insert("end", eq + "\n", "eq")
                t.insert("end", "in: " + where + "\n", "where")
        t.configure(state="disabled")

    def jump(self, _=None):
        sel = self.lb.curselection()
        if sel:
            self.txt.yview(f"sec{sel[0]}")

    def launch(self):
        """Run the engine script as its own program.  This window stays open and does not depend on it."""
        if not os.path.exists(ENGINE):
            messagebox.showerror("Launch engine", f"physics_engine.py was not found next to this menu:\n{ENGINE}")
            return
        try:
            p = subprocess.Popen([sys.executable, ENGINE, *ENGINE_ARGS], cwd=HERE)
        except OSError as e:
            messagebox.showerror("Launch engine", f"Could not start the engine:\n{e}")
            return
        self.status.set(f"Engine started (process {p.pid})")
        self.launch_b.configure(state="disabled")                # no double-click starting two engines
        self.root.after(2500, lambda: self.launch_b.configure(state="normal"))


if __name__ == "__main__":
    root = tk.Tk()
    Menu(root)
    root.mainloop()
