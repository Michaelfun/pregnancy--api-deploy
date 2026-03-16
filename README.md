# Pregnancy Vitals API - Simple Usage Guide

## Quick Start

**API Base URL:** `https://pregnancy-es8z.onrender.com`

No authentication required - just send requests!

---

## Available Endpoints

### 1. Check if API is Working
**GET** `/health`

**Example:**
```powershell
(Invoke-WebRequest -Uri "https://pregnancy-es8z.onrender.com/health" -Method GET).Content
```

**Response:**
```json
{
  "status": "healthy",
  "model": "pregnancy_vitals_model",
  "features": ["pulse", "respiration", "temperature", "systolic", "diastolic", "oxygen"]
}
```

---

### 2. Get Normal Ranges
**GET** `/ranges`

See what values are considered normal for each vital sign.

**Example:**
```powershell
(Invoke-WebRequest -Uri "https://pregnancy-es8z.onrender.com/ranges" -Method GET).Content
```

**Response:**
```json
{
  "normal_ranges": {
    "pulse": {"min": 60, "max": 100},
    "respiration": {"min": 12, "max": 20},
    "temperature": {"min": 35, "max": 38},
    "systolic": {"min": 80, "max": 120},
    "diastolic": {"min": 80, "max": 120},
    "oxygen": {"min": 95, "max": 100}
  }
}
```

---

### 3. Predict Risk Level (Single Patient)
**POST** `/predict`

Analyze one set of vital signs and get risk assessment.

**Request:**
```powershell
$body = @{
  pulse       = 75
  respiration = 16
  temperature = 36.5
  systolic    = 120
  diastolic   = 80
  oxygen      = 98
} | ConvertTo-Json

(Invoke-WebRequest `
  -Uri "https://pregnancy-es8z.onrender.com/predict" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body $body
).Content
```

**What to Send:**
```json
{
  "pulse": 75,           // Heart rate (beats per minute)
  "respiration": 16,     // Breathing rate (breaths per minute)
  "temperature": 36.5,   // Body temperature (°C)
  "systolic": 120,       // Systolic blood pressure (mmHg)
  "diastolic": 80,       // Diastolic blood pressure (mmHg)
  "oxygen": 98           // Oxygen saturation (%)
}
```

**Response:**
```json
{
  "risk_level": "normal",
  "risk_probabilities": {
    "normal": 0.85,
    "low": 0.10,
    "medium": 0.04,
    "high": 0.01
  },
  "vital_signs_analysis": {
    "pulse": {
      "value": 75,
      "status": "normal",
      "normal_range": "60-100",
      "factor_risk": "normal",
      "factor_risk_probabilities": {
        "normal": 0.95,
        "high": 0.05
      }
    }
    // ... similar analysis for other vital signs
  }
}
```

**Risk Levels:**
- `normal` - All values within normal range
- `low` - Minor deviations
- `medium` - Moderate concerns
- `high` - Significant concerns requiring attention

---

### 4. Batch Predict (Multiple Patients)
**POST** `/batch_predict`

Analyze multiple sets of vital signs at once.

**Request:**
```powershell
$body = @(
  @{
    pulse       = 75
    respiration = 16
    temperature = 36.5
    systolic    = 120
    diastolic   = 80
    oxygen      = 98
  },
  @{
    pulse       = 110
    respiration = 24
    temperature = 38.5
    systolic    = 150
    diastolic   = 100
    oxygen      = 91
  }
) | ConvertTo-Json

(Invoke-WebRequest `
  -Uri "https://pregnancy-es8z.onrender.com/batch_predict" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body $body
).Content
```

**What to Send:** Array of vital signs objects (same format as single predict)

**Response:**
```json
{
  "risk_levels": ["normal", "high"],
  "count": 2,
  "detailed_results": [
    {
      "record": { /* original input */ },
      "risk_level": "normal",
      "vital_signs_analysis": { /* detailed analysis */ }
    },
    {
      "record": { /* original input */ },
      "risk_level": "high",
      "vital_signs_analysis": { /* detailed analysis */ }
    }
  ]
}
```

---

## Code Examples

### JavaScript (Fetch API)
```javascript
// Single prediction
const response = await fetch('https://pregnancy-es8z.onrender.com/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    pulse: 75,
    respiration: 16,
    temperature: 36.5,
    systolic: 120,
    diastolic: 80,
    oxygen: 98
  })
});

const result = await response.json();
console.log('Risk Level:', result.risk_level);
console.log('Probabilities:', result.risk_probabilities);
```

### Python (requests library)
```python
import requests

url = "https://pregnancy-es8z.onrender.com/predict"
data = {
    "pulse": 75,
    "respiration": 16,
    "temperature": 36.5,
    "systolic": 120,
    "diastolic": 80,
    "oxygen": 98
}

response = requests.post(url, json=data)
result = response.json()

print(f"Risk Level: {result['risk_level']}")
print(f"Probabilities: {result['risk_probabilities']}")
```

### Python (Batch)
```python
import requests

url = "https://pregnancy-es8z.onrender.com/batch_predict"
data = [
    {
        "pulse": 75,
        "respiration": 16,
        "temperature": 36.5,
        "systolic": 120,
        "diastolic": 80,
        "oxygen": 98
    },
    {
        "pulse": 110,
        "respiration": 24,
        "temperature": 38.5,
        "systolic": 150,
        "diastolic": 100,
        "oxygen": 91
    }
]

response = requests.post(url, json=data)
result = response.json()

for i, risk in enumerate(result['risk_levels']):
    print(f"Patient {i+1}: {risk}")
```

---

## Required Fields

All prediction endpoints require these 6 vital signs:

| Field | Description | Example Value | Unit |
|-------|-------------|---------------|------|
| `pulse` | Heart rate | 75 | beats/min |
| `respiration` | Breathing rate | 16 | breaths/min |
| `temperature` | Body temperature | 36.5 | °C |
| `systolic` | Systolic BP | 120 | mmHg |
| `diastolic` | Diastolic BP | 80 | mmHg |
| `oxygen` | Oxygen saturation | 98 | % |

---

## Example Scenarios

### Scenario 1: Normal Vitals
```json
{
  "pulse": 72,
  "respiration": 15,
  "temperature": 36.6,
  "systolic": 115,
  "diastolic": 78,
  "oxygen": 98
}
```
**Expected Result:** `risk_level: "normal"`

### Scenario 2: Elevated Vitals (Higher Risk)
```json
{
  "pulse": 110,
  "respiration": 24,
  "temperature": 38.5,
  "systolic": 150,
  "diastolic": 100,
  "oxygen": 91
}
```
**Expected Result:** `risk_level: "high"` or `"medium"`

### Scenario 3: Low Vitals
```json
{
  "pulse": 55,
  "respiration": 10,
  "temperature": 35.2,
  "systolic": 70,
  "diastolic": 50,
  "oxygen": 93
}
```
**Expected Result:** `risk_level: "high"` or `"medium"`

---

## Common Errors

### Error 400: Bad Request
**Cause:** Missing required fields or invalid data types
**Solution:** Make sure all 6 vital signs are included and are numbers (not strings)

**PowerShell: print the 400 response body (very useful on Render):**
```powershell
try {
  $body = @{
    pulse       = 75
    respiration = 16
    temperature = 36.5
    systolic    = 120
    diastolic   = 80
    oxygen      = 98
  } | ConvertTo-Json

  Invoke-WebRequest `
    -Uri "https://pregnancy-es8z.onrender.com/predict" `
    -Method POST `
    -Headers @{ "Content-Type" = "application/json" } `
    -Body $body `
    | Out-Null
}
catch {
  "HTTP Status: " + $_.Exception.Response.StatusCode.value__
  $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
  $reader.ReadToEnd()
}
```

**Example Error:**
```json
{
  "error": "Missing required field: pulse"
}
```

### Error 500: Server Error
**Cause:** API server issue
**Solution:** Check `/health` endpoint first. If healthy, try again in a few moments.

---

## Testing Tips

1. **Start with Health Check** - Always verify API is running first
2. **Use Normal Values First** - Test with values within normal ranges
3. **Check Response Structure** - Understand what data you'll receive
4. **Handle Errors** - Always check for error responses in your code

---

## Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/ranges` | GET | Get normal ranges |
| `/predict` | POST | Single prediction |
| `/batch_predict` | POST | Multiple predictions |

**Base URL:** `https://pregnancy-es8z.onrender.com`

---

## Need Help?
- Verify API is running: `GET /health`
