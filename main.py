

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from ticket_class import ai_ticket_classifier

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request
    )


@app.post("/classify", response_class=HTMLResponse)
async def classify(ticket: str = Form(...)):

    result = ai_ticket_classifier(ticket)

    return f"""
<div class="p-8">

    <div class="flex items-center justify-between mb-8">

        <h2 class="text-xl font-semibold text-slate-900">
            Ticket Analysis
        </h2>

        <span class="px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-sm">
            Completed
        </span>

    </div>

    <div class="grid grid-cols-2 gap-4 mb-6">

        <div class="border rounded-xl p-4">
            <p class="text-xs text-slate-500">Category</p>
            <p class="font-semibold">{result.category}</p>
        </div>

        <div class="border rounded-xl p-4">
            <p class="text-xs text-slate-500">Priority</p>
            <p class="font-semibold">{result.priority}</p>
        </div>

    </div>

    <div class="border rounded-xl p-4 mb-4">

        <p class="text-xs text-slate-500 mb-2">
            Escalation Required
        </p>

        <p class="font-semibold">
            {"Yes" if result.is_escalted else "No"}
        </p>

    </div>

    <div class="border rounded-xl p-4">

        <p class="text-xs text-slate-500 mb-2">
            Summary
        </p>

        <p class="text-slate-700 leading-relaxed">
            {result.summary}
        </p>

    </div>

</div>
"""