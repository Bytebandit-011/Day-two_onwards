# ☕ Murf AI Voice Agent – Day 2: Coffee Shop Barista

This repository contains my submission for **Day 2** of the  
**Murf AI Voice Agents Challenge**, where I transformed the basic agent into a fully interactive **Coffee Shop Barista**.

---

## 🎯 Primary Goal (Required)

### ✅ Persona  
I turned the agent into a friendly barista for the coffee brand **YourCoffeeCo** (can be changed).  
The agent speaks warmly, guides the user, and confirms order details.

---

## 🛠️ Order State System

The agent maintains a structured order object:

```json
{
  "drinkType": "",
  "size": "",
  "milk": "",
  "extras": [],
  "name": ""
}