#!/usr/bin/env python3
"""Generate figures ONLY from measured run_summary.json, never from demo numbers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from task_arithmetic.config import RESULTS


def get(bucket, variant, task, split="test"):
    return next(x for x in bucket["rows"] if x["variant"]==variant and x["task"]==task and x["split"]==split)["accuracy"]


def plot_negation(bucket, out):
    xs = [0.0,0.25,0.5,1.0]
    fig, ax = plt.subplots(figsize=(7,4.5))
    for task, label in [("amazon_sent","Target: Amazon sentiment"),("agnews_topic","Control: AG News")]:
        ys = [get(bucket, f"neg_alpha_{a:g}", task) for a in xs]
        ax.plot(xs, ys, marker="o", label=label)
    ax.axvline(0.5, linestyle=":", linewidth=1, label="Predeclared alpha=0.5")
    ax.set(title="Negation: remove Amazon sentiment task direction", xlabel="Negation coefficient alpha",
           ylabel="Constrained-label accuracy", ylim=(0,1), xticks=xs)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout();fig.savefig(out,dpi=175);plt.close(fig)


def plot_addition(bucket, out):
    labels = ["Base", "Amazon FT", "AG News FT", "Sum (selected)"]
    variants = ["base","amazon_finetuned","agnews_finetuned","addition_selected"]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8,4.5))
    for offset,task,label in [(-0.19,"amazon_sent","Amazon sentiment"),(0.19,"agnews_topic","AG News")]:
        ys = [get(bucket,v,task) for v in variants]
        ax.bar([i+offset for i in x], ys, width=0.36, label=label)
    ax.set(title=f"Addition: selected alpha={bucket['selected_alpha_from_validation']}", ylabel="Accuracy",
           xticks=x, xticklabels=labels, ylim=(0,1))
    ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(out,dpi=175);plt.close(fig)


def plot_analogy(bucket, out):
    xs = [0.0,0.1,0.25,0.5]
    yelp = [get(bucket,f"analogy_domain_{a:g}","yelp_sent") for a in xs]
    amazon = [get(bucket,f"analogy_domain_{a:g}","amazon_sent") for a in xs]
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(xs,yelp,marker="o",label="Target: Yelp sentiment (test only)")
    ax.plot(xs,amazon,marker="s",label="Source: Amazon sentiment")
    ax.axvline(0.25,linestyle=":",linewidth=1,label="Predeclared alpha=0.25")
    ax.set(title="Analogy: Amazon sentiment + Yelp LM - Amazon LM",
           xlabel="Domain-shift coefficient",ylabel="Accuracy",ylim=(0,1),xticks=xs)
    ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out,dpi=175);plt.close(fig)


def main():
    data = json.loads((RESULTS / "run_summary.json").read_text())
    path = RESULTS / "figures"
    path.mkdir(parents=True,exist_ok=True)
    methods=data["experiments"]
    for name,fun in [("negation",plot_negation),("addition",plot_addition),("analogy",plot_analogy)]:
        out = path / f"{name}.png"
        fun(methods[name],out)
        print(out)

if __name__=="__main__":main()
