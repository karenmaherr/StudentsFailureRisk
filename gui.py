import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import os ,sys

def resource_path(filename):
    base = getattr(sys, "_MEIPASS",os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)
model= joblib.load(resource_path("best_model.pkl"))
encoder= joblib.load(resource_path("encoder.pkl"))
scaler= joblib.load(resource_path("scaler.pkl"))
root= tk.Tk()
root.title("Academic Failure Risk")
root.geometry("480x600")
root.configure(bg="white")
root.resizable(False, False)
FONT= ("Segoe UI",11)
FONT_BOLD= ("Segoe UI", 13, "bold")
tk.Label(root,text="Academic Failure Risk", font=("Segoe UI",16, "bold"),
bg="white").pack()
form =tk.Frame(root, bg="white")
form.pack()
def add_slider(text,frm,to,res=1):
    tk.Label(form, text=text, font=FONT, bg="white").pack()
    s= tk.Scale(form, from_=frm, to=to, resolution=res, orient="horizontal",
    length=350, bg="white", highlightthickness=0)
    s.pack()
    return s
social= add_slider("Daily social media hours", 0, 24, 0.5)
sleep= add_slider("Sleep hours", 0, 24, 0.5)
mental = add_slider("How do you score your mental health? (0-100)", 0, 100)
tk.Label(form, text="How much do you feel Burnedout?", font=FONT, bg="white").pack()
burnout= ttk.Combobox(form, values=["Low","Moderate","High" ,"Severe"],state="readonly", width=20)
burnout.set("Low")
burnout.pack()
academic= add_slider("Your average Academic performance score (0-100)", 0, 100)
result= tk.Label(root, text="", font=FONT_BOLD, bg="white")
result.pack()
def predict():
    row = pd.DataFrame([[
        social.get(), sleep.get(), mental.get(), burnout.get(), academic.get()]],columns=[
        "Daily_Social_Media_Hours", "Sleep_Hours", "Mental_Health_Score",
        "Burnout_Level", "Academic_Performance_Score"])
    cat_cols = row.select_dtypes(include="object").columns
    row[cat_cols] = encoder.transform(row[cat_cols])
    row_scaled = scaler.transform(row)
    pred= model.predict(row_scaled)[0]
    proba = model.predict_proba(row_scaled)[0]
    if pred== 1:
        result.config(
            text=f"At risk — failure probability: {proba[1]*100:.1f}%",
            fg="red")
    else:
        result.config(
            text=f"Low risk — success probability: {proba[0]*100:.1f}%",
            fg="green")
tk.Button(root, text="Predict", font=FONT_BOLD, command=predict,
bg="purple", fg="white", width=18, relief="flat").pack()
root.mainloop()
