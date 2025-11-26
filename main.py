from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import uvicorn
from pydantic import BaseModel, EmailStr
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from dotenv import load_dotenv
import os
import google.generativeai as genai
from contextlib import contextmanager  # ← ADD THIS

load_dotenv() 

import psycopg2
import pandas as pd

# ============ DATABASE CONNECTION MANAGEMENT ============

def get_db_connection():
    """Create a new database connection"""
    return psycopg2.connect(
        host="aws-0-ap-southeast-1.pooler.supabase.com",
        port=5432,
        database="postgres",
        user="postgres.mnxcvpzefuwuccprqnxl",
        password=os.getenv("PASS_DB"),
        sslmode="require"
    )

@contextmanager
def get_db():
    """Context manager for database connections - automatically closes connection"""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def query_to_dataframe(conn, query):
    """Execute query and return results as DataFrame"""
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=colnames)
        cur.close()
        return df
    except Exception as e:
        print("Error executing query:", e)
        return pd.DataFrame()

# ============ EMAIL & FILE LOADING ============

sender_email = "ahmadimanh3@gmail.com"
app_password = os.getenv("APP_PASSWORD")
smtp_server = "smtp.gmail.com"
port = 587

top_10_products_ts = pd.read_csv(os.path.join("csv_files", "top10_product_ts.csv"))
top_10_products_forecast = pd.read_csv(os.path.join("csv_files", "top10_products_forecast_results.csv"))
forecasted_orders = pd.read_csv(os.path.join("csv_files", "forecasted_orders_synthetic_1.csv"))
orders_weekly = pd.read_csv(os.path.join("csv_files", "synthetic_orders_1.csv"))

# ============ FASTAPI APP SETUP ============

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============ MODELS ============

class EmailRequest(BaseModel):
    email: EmailStr
    item: str
    quantity: int

# ============ EMAIL ENDPOINT ============

@app.post("/send-email-stock-alert")
async def send_stock_email(request: EmailRequest) -> bool:
    receiver_email = request.email
    subject = "Request to buy more stocks"

    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2E86C1;">Stock Refill Request</h2>
        <p>Dear Supplier,</p>
        <p>We would like to request a restock for the following item:</p>
        <table style="border-collapse: collapse; margin: 20px 0;">
        <tr>
            <td style="padding: 8px; font-weight: bold;">Item:</td>
            <td style="padding: 8px;">{request.item}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">Quantity:</td>
            <td style="padding: 8px;">{request.quantity}</td>
        </tr>
        </table>
        <p>Please process this request as soon as possible. Thank you!</p>
        <p>Best regards,<br>AdventureWorks</p>
    </body>
    </html>
    """

    message = MIMEMultipart("alternative")
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        return True
    except Exception as e:
        print(f"Email send error: {e}")
        return False

# ============ PAGE ROUTES ============

@app.get("/", response_class=HTMLResponse)
async def analytics_home(request: Request):
    """Serve the analytics dashboard"""
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/main_page", response_class=HTMLResponse)
async def serve_step1_business_info(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/details_page", response_class=HTMLResponse)
async def serve_details(request: Request):
    return templates.TemplateResponse("details.html", {"request": request})

@app.get("/orders_page", response_class=HTMLResponse)
async def serve_orders(request: Request):
    return templates.TemplateResponse("orders.html", {"request": request})

@app.get("/impact_metrics", response_class=HTMLResponse)
async def serve_impact_metrics(request: Request):
    """Serve the business impact metrics dashboard"""
    return templates.TemplateResponse("impact_metrics.html", {"request": request})

# ============ INVENTORY ENDPOINTS ============

@app.get("/product-change-status")
async def product_change_status(product: str):
    return increase_or_decrease(product)

def increase_or_decrease(product):
    """Calculate if product demand is increasing, decreasing, or stable"""
    sales_data = top_10_products_ts
    forecast_data = top_10_products_forecast
    sales_data = sales_data[sales_data["ProductName"] == product]
    forecast_data = forecast_data[forecast_data["Product"] == product]
    sales_data = sales_data.iloc[:-1]
    last_4_weeks_sale = sum(sales_data.iloc[-4:]["sum"])
    forecast_lower_limit = sum(forecast_data.iloc[:4]["Lower 80%"])
    forecast_upper_limit = sum(forecast_data.iloc[:4]["Upper 80%"])
    prediction_status = "Decrease"
    if (last_4_weeks_sale < forecast_lower_limit):
        prediction_status = "Increase"
    elif (last_4_weeks_sale < forecast_upper_limit and last_4_weeks_sale > forecast_lower_limit):
        prediction_status = "No significant changes"
    else:
        prediction_status = "Decrease"
    
    return_dict = {
        "last_4_weeks_sale": last_4_weeks_sale,
        "forecast_lower_limit": forecast_lower_limit,
        "forecast_upper_limit": forecast_upper_limit,
        "prediction_status": prediction_status
    }

    return return_dict

@app.get("/get_unique_products")
async def unique_products():
    sales_data = top_10_products_ts
    return sales_data["ProductName"].unique().tolist()

@app.get("/restock_or_not")
async def restock_or_not(product):
    """Check if product needs restocking based on predicted demand"""
    try:
        with get_db() as conn:  # ← FIXED: Using context manager
            query = """SELECT * FROM "stock_table" """
            stock_table = query_to_dataframe(conn, query)
            
            # Check if product exists
            product_rows = stock_table[stock_table["Products"] == product]
            if product_rows.empty:
                raise HTTPException(status_code=404, detail=f"Product '{product}' not found")
            
            stocks = int(product_rows.iloc[0]["Stock"])
            upper_limit = round(float(increase_or_decrease(product)["forecast_upper_limit"]), 2)
            lower_limit = round(float(increase_or_decrease(product)["forecast_lower_limit"]), 2)
            
            restock = False
            if stocks < upper_limit:
                restock = True

            return {
                "Stocks": int(stocks),
                "Upper_Limit": float(upper_limit),
                "Predictions": f"{float(lower_limit)} - {float(upper_limit)}",
                "need_restock": bool(restock)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in restock_or_not for {product}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/linegraph_endpoint")
async def linegraph_endpoint(product: str):
    historical_data = top_10_products_ts[top_10_products_ts["ProductName"] == product]
    historical_data = historical_data.iloc[:-1]
    forecasted_data = top_10_products_forecast[top_10_products_forecast["Product"] == product]

    return {
        "historical_data": historical_data.to_dict(orient="records"),
        "forecasted_data": forecasted_data.to_dict(orient="records")
    }

@app.get("/linegraph_endpoint_orders")
async def linegraph_endpoint_orders():
    historical_data = orders_weekly 
    historical_data = historical_data.iloc[:-1]
    forecasted_data = forecasted_orders

    return {
        "historical_data": historical_data.to_dict(orient="records"),
        "forecasted_data": forecasted_data.to_dict(orient="records")
    }

@app.get("/adjust_schedule")
async def adjust_schedule():
    staff_names = [
        "Alicia Tan", "Jason Koh", "Bryan Lim", "Chloe Wong", "Farah Aziz",
        "Daniel Lee", "Natalie Cheng", "Amir Hakim", "Stephanie Ong", "Kelvin Yap"
    ]

    df_schedule = pd.read_csv(os.path.join("csv_files", "Staff_schedule_nextweek.csv"))

    prompt = f"""
    This is the staff schedule for next week
    {df_schedule}

    This is the staff list:
    {staff_names}

    It seems like the shop is getting busier next week, I need you to help me reschedule by adding more staff at each session.
    Your reply should only consist of Python code that creates the new schedule in a pandas DataFrame.
    The final dataframe should be stored in a variable called final_df.
    """

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)

    code_str = response.text.replace("```python", "").replace("```", "").strip()

    namespace = {"pd": pd, "staff_names": staff_names, "df_schedule": df_schedule}

    exec(code_str, namespace)

    final_df = namespace.get("final_df")

    return {"schedule": final_df.to_dict(orient="records")}

# ============ ANALYTICS ENDPOINTS ============

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get overview KPIs with period comparison"""
    try:
        with get_db() as conn:  # ← FIXED: Using context manager
            # Get max date from data
            date_query = """
            SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date
            FROM "Sales_data"
            """
            df_date = pd.read_sql(date_query, conn)
            max_date = df_date['max_date'].iloc[0]
            
            # Current period (last 30 days of available data)
            query_current = f"""
            SELECT 
                COUNT(DISTINCT "OrderNumber") as total_orders,
                COALESCE(SUM("OrderQuantity"), 0) as total_units,
                COUNT(DISTINCT "CustomerKey") as unique_customers
            FROM "Sales_data"
            WHERE TO_DATE("OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '30 days'
            """
            
            df_current = pd.read_sql(query_current, conn)
            
            # Previous period (30-60 days before max date)
            query_previous = f"""
            SELECT 
                COUNT(DISTINCT "OrderNumber") as total_orders,
                COALESCE(SUM("OrderQuantity"), 0) as total_units,
                COUNT(DISTINCT "CustomerKey") as unique_customers
            FROM "Sales_data"
            WHERE TO_DATE("OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '60 days'
            AND TO_DATE("OrderDate", 'DD/MM/YYYY') < '{max_date}'::date - INTERVAL '30 days'
            """
            
            df_previous = pd.read_sql(query_previous, conn)
            
            current = {
                "total_orders": int(df_current['total_orders'].iloc[0] or 0),
                "total_units": int(df_current['total_units'].iloc[0] or 0),
                "unique_customers": int(df_current['unique_customers'].iloc[0] or 0)
            }
            
            previous = {
                "total_orders": int(df_previous['total_orders'].iloc[0] or 0),
                "total_units": int(df_previous['total_units'].iloc[0] or 0),
                "unique_customers": int(df_previous['unique_customers'].iloc[0] or 0)
            }
            
            def safe_change(curr, prev):
                if prev == 0:
                    return 100.0 if curr > 0 else 0.0
                return round(((curr - prev) / prev) * 100, 2)
            
            changes = {
                "orders": safe_change(current["total_orders"], previous["total_orders"]),
                "units": safe_change(current["total_units"], previous["total_units"]),
                "customers": safe_change(current["unique_customers"], previous["unique_customers"])
            }
            
            return {
                "current_period": current,
                "previous_period": previous,
                "changes": changes
            }
        
    except Exception as e:
        print(f"Error in overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/sales-trend")
async def get_sales_trend(weeks: int = 12):
    """Get weekly sales trend"""
    try:
        with get_db() as conn:  # ← FIXED: Using context manager
            # Get max date
            date_query = """SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date FROM "Sales_data" """
            df_date = pd.read_sql(date_query, conn)
            max_date = df_date['max_date'].iloc[0]
            
            query = f"""
            SELECT 
                DATE_TRUNC('week', TO_DATE("OrderDate", 'DD/MM/YYYY')) as week_start,
                SUM("OrderQuantity") as total_units,
                COUNT(DISTINCT "OrderNumber") as order_count
            FROM "Sales_data"
            WHERE TO_DATE("OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '{weeks} weeks'
            GROUP BY week_start
            ORDER BY week_start
            """
            
            df = pd.read_sql(query, conn)
            
            trend_data = []
            for _, row in df.iterrows():
                trend_data.append({
                    "week": row['week_start'].strftime('%Y-%m-%d'),
                    "units": int(row['total_units']),
                    "orders": int(row['order_count'])
                })
            
            return {
                "data": trend_data,
                "total_weeks": len(trend_data)
            }
        
    except Exception as e:
        print(f"Error in sales trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/top-products")
async def get_top_products(limit: int = 10):
    """Get top selling products"""
    try:
        with get_db() as conn:  # ← FIXED: Using context manager
            # Get max date
            date_query = """SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date FROM "Sales_data" """
            df_date = pd.read_sql(date_query, conn)
            max_date = df_date['max_date'].iloc[0]
            
            query = f"""
            SELECT 
                p."ProductName",
                COALESCE(pc."CategoryName", 'Uncategorized') as category_name,
                SUM(s."OrderQuantity") as total_sold,
                COUNT(DISTINCT s."OrderNumber") as order_frequency
            FROM "Sales_data" s
            JOIN "Product_Lookup" p ON s."ProductKey" = p."ProductKey"
            LEFT JOIN "Product_Subcategories_Lookup" psc ON p."ProductSubcategoryKey" = psc."ProductSubcategoryKey"
            LEFT JOIN "Product_Category_Lookup" pc ON psc."ProductCategoryKey" = pc."ProductCategoryKey"
            WHERE TO_DATE(s."OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '30 days'
            GROUP BY p."ProductName", pc."CategoryName"
            ORDER BY total_sold DESC
            LIMIT {limit}
            """
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return {"products": [], "total": 0}
            
            products = []
            for _, row in df.iterrows():
                products.append({
                    "product": str(row['ProductName']) if pd.notna(row['ProductName']) else 'Unknown',
                    "category": str(row['category_name']) if pd.notna(row['category_name']) else 'Uncategorized',
                    "units_sold": int(row['total_sold']) if pd.notna(row['total_sold']) else 0,
                    "orders": int(row['order_frequency']) if pd.notna(row['order_frequency']) else 0
                })
            
            return {
                "products": products,
                "total": len(products)
            }
        
    except Exception as e:
        print(f"Error in top products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/category-breakdown")
async def get_category_breakdown():
    """Get sales distribution by category"""
    try:
        with get_db() as conn:  # ← FIXED: Using context manager
            # Get max date
            date_query = """SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date FROM "Sales_data" """
            df_date = pd.read_sql(date_query, conn)
            max_date = df_date['max_date'].iloc[0]
            
            query = f"""
            SELECT 
                COALESCE(pc."CategoryName", 'Uncategorized') as category_name,
                SUM(s."OrderQuantity") as total_units,
                COUNT(DISTINCT s."OrderNumber") as order_count,
                COUNT(DISTINCT s."ProductKey") as product_count
            FROM "Sales_data" s
            JOIN "Product_Lookup" p ON s."ProductKey" = p."ProductKey"
            LEFT JOIN "Product_Subcategories_Lookup" psc ON p."ProductSubcategoryKey" = psc."ProductSubcategoryKey"
            LEFT JOIN "Product_Category_Lookup" pc ON psc."ProductCategoryKey" = pc."ProductCategoryKey"
            WHERE TO_DATE(s."OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '30 days'
            GROUP BY pc."CategoryName"
            ORDER BY total_units DESC
            """
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return {"categories": [], "total_units": 0}
            
            total_units = df['total_units'].sum()
            
            categories = []
            for _, row in df.iterrows():
                categories.append({
                    "category": str(row['category_name']) if pd.notna(row['category_name']) else 'Uncategorized',
                    "units": int(row['total_units']) if pd.notna(row['total_units']) else 0,
                    "orders": int(row['order_count']) if pd.notna(row['order_count']) else 0,
                    "products": int(row['product_count']) if pd.notna(row['product_count']) else 0,
                    "percentage": round((row['total_units'] / total_units) * 100, 2) if total_units > 0 else 0
                })
            
            return {
                "categories": categories,
                "total_units": int(total_units) if pd.notna(total_units) else 0
            }
        
    except Exception as e:
        print(f"Error in category breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/territory-performance")
async def get_territory_performance(limit: int = 10):
    """Get sales by territory - aggregated by country"""
    try:
        with get_db() as conn:  # ← FIXED: Using context manager
            # Get max date
            date_query = """SELECT MAX(TO_DATE("OrderDate", 'DD/MM/YYYY')) as max_date FROM "Sales_data" """
            df_date = pd.read_sql(date_query, conn)
            max_date = df_date['max_date'].iloc[0]
            
            query = f"""
            SELECT 
                t."Country",
                SUM(s."OrderQuantity") as total_units,
                COUNT(DISTINCT s."OrderNumber") as order_count
            FROM "Sales_data" s
            JOIN "Territory_Lookup" t ON s."TerritoryKey" = t."SalesTerritoryKey"
            WHERE TO_DATE(s."OrderDate", 'DD/MM/YYYY') >= '{max_date}'::date - INTERVAL '30 days'
            GROUP BY t."Country"
            ORDER BY total_units DESC
            LIMIT {limit}
            """
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                return {"territories": [], "total": 0, "total_units": 0}
            
            total_units = df['total_units'].sum()
            
            territories = []
            for _, row in df.iterrows():
                territories.append({
                    "country": str(row['Country']) if pd.notna(row['Country']) else 'Unknown',
                    "units": int(row['total_units']) if pd.notna(row['total_units']) else 0,
                    "orders": int(row['order_count']) if pd.notna(row['order_count']) else 0,
                    "percentage": round((row['total_units'] / total_units) * 100, 2) if total_units > 0 else 0
                })
            
            return {
                "territories": territories,
                "total": len(territories),
                "total_units": int(total_units) if pd.notna(total_units) else 0
            }
        
    except Exception as e:
        print(f"Error in territory performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ BUSINESS IMPACT METRICS ENDPOINTS ============

# ============ BUSINESS IMPACT METRICS ENDPOINTS (WITH DYNAMIC PRICING) ============

@app.get("/api/metrics/stockout-prevention")
async def calculate_stockout_prevention():
    """Calculate stockout prevention metrics using actual product prices"""
    try:
        with get_db() as conn:
            # Get stock table with actual product prices
            query = """
            SELECT 
                s.*,
                COALESCE(p."ProductPrice", 50) as unit_price
            FROM "stock_table" s
            LEFT JOIN "Product_Lookup" p ON s."Products" = p."ProductName"
            """
            stock_table = query_to_dataframe(conn, query)
            
            products = stock_table["Products"].unique()
            
            total_products = len(products)
            products_at_risk = 0
            prevented_stockouts = 0
            potential_lost_revenue = 0
            
            for product in products:
                stock_info = await restock_or_not(product)
                
                # Count products that would stockout without forecast
                if stock_info["need_restock"]:
                    products_at_risk += 1
                    
                    # Calculate potential lost sales using ACTUAL product price
                    upper_limit = stock_info["Upper_Limit"]
                    current_stock = stock_info["Stocks"]
                    shortfall = max(0, upper_limit - current_stock)
                    
                    # Get actual unit price for this product
                    product_row = stock_table[stock_table["Products"] == product]
                    if not product_row.empty:
                        unit_price = float(product_row.iloc[0]["unit_price"])
                    else:
                        unit_price = 50  # Fallback if price not found
                    
                    potential_lost_revenue += shortfall * unit_price
                    
                    # If we're alerting now, we're preventing the stockout
                    prevented_stockouts += 1
            
            stockout_rate_without_forecast = (products_at_risk / total_products) * 100 if total_products > 0 else 0
            stockout_rate_with_forecast = max(0, (products_at_risk - prevented_stockouts) / total_products * 100) if total_products > 0 else 0
            
            return {
                "total_products_tracked": total_products,
                "products_at_risk": products_at_risk,
                "stockouts_prevented": prevented_stockouts,
                "stockout_rate_without_forecast": round(stockout_rate_without_forecast, 2),
                "stockout_rate_with_forecast": round(stockout_rate_with_forecast, 2),
                "improvement_percentage": round(stockout_rate_without_forecast - stockout_rate_with_forecast, 2),
                "potential_lost_revenue_prevented": round(potential_lost_revenue, 2)
            }
        
    except Exception as e:
        print(f"Error in stockout prevention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/inventory-efficiency")
async def calculate_inventory_efficiency():
    """Calculate inventory optimization metrics using actual product costs"""
    try:
        with get_db() as conn:
            # Get stock table with actual product costs
            # Using ProductCost if available, otherwise estimate as 60% of ProductPrice
            query = """
            SELECT 
                s.*,
                COALESCE(p."ProductCost", p."ProductPrice" * 0.6, 30) as unit_cost
            FROM "stock_table" s
            LEFT JOIN "Product_Lookup" p ON s."Products" = p."ProductName"
            """
            stock_table = query_to_dataframe(conn, query)
            
            products = stock_table["Products"].unique()
            
            total_excess_inventory = 0
            total_excess_inventory_value = 0
            total_optimal_inventory = 0
            overstocked_products = 0
            understocked_products = 0
            optimal_products = 0
            
            for product in products:
                stock_info = await restock_or_not(product)
                forecast_info = increase_or_decrease(product)
                
                current_stock = stock_info["Stocks"]
                upper_limit = stock_info["Upper_Limit"]
                lower_limit = forecast_info["forecast_lower_limit"]
                
                # Get actual unit cost for this product
                product_row = stock_table[stock_table["Products"] == product]
                if not product_row.empty:
                    unit_cost = float(product_row.iloc[0]["unit_cost"])
                else:
                    unit_cost = 30  # Fallback
                
                # Calculate optimal inventory (add 10% safety stock)
                optimal_stock = upper_limit * 1.1
                total_optimal_inventory += optimal_stock
                
                # Calculate excess
                if current_stock > upper_limit:
                    excess = current_stock - upper_limit
                    total_excess_inventory += excess
                    total_excess_inventory_value += excess * unit_cost
                    overstocked_products += 1
                elif current_stock < lower_limit:
                    understocked_products += 1
                else:
                    optimal_products += 1
            
            # Carrying cost is typically 20-25% annually
            annual_carrying_cost_saved = total_excess_inventory_value * 0.25
            monthly_carrying_cost_saved = annual_carrying_cost_saved / 12
            
            total_products = len(products)
            overstock_rate = (overstocked_products / total_products) * 100 if total_products > 0 else 0
            understock_rate = (understocked_products / total_products) * 100 if total_products > 0 else 0
            optimal_rate = (optimal_products / total_products) * 100 if total_products > 0 else 0
            
            return {
                "total_products": total_products,
                "overstocked_products": overstocked_products,
                "understocked_products": understocked_products,
                "optimal_products": optimal_products,
                "overstock_rate": round(overstock_rate, 2),
                "understock_rate": round(understock_rate, 2),
                "optimal_rate": round(optimal_rate, 2),
                "excess_inventory_units": round(total_excess_inventory, 2),
                "excess_inventory_value": round(total_excess_inventory_value, 2),
                "monthly_carrying_cost_saved": round(monthly_carrying_cost_saved, 2),
                "annual_carrying_cost_saved": round(annual_carrying_cost_saved, 2)
            }
        
    except Exception as e:
        print(f"Error in inventory efficiency: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/forecast-accuracy")
async def calculate_forecast_accuracy():
    """Calculate forecasting accuracy metrics"""
    try:
        sales_data = top_10_products_ts
        forecast_data = top_10_products_forecast
        
        products = sales_data["ProductName"].unique()
        
        total_mape = 0
        accurate_forecasts = 0
        products_evaluated = 0
        
        for product in products:
            product_sales = sales_data[sales_data["ProductName"] == product]
            product_forecast = forecast_data[forecast_data["Product"] == product]
            
            if len(product_sales) > 0 and len(product_forecast) > 0:
                # Get last 4 weeks actual vs forecast
                actual_last_4 = product_sales.iloc[-5:-1]["sum"].values
                forecast_median = product_forecast.iloc[:4]["Median Forecast"].values
                
                if len(actual_last_4) == len(forecast_median) and len(actual_last_4) > 0:
                    products_evaluated += 1
                    # Calculate MAPE
                    mape = np.mean(np.abs((actual_last_4 - forecast_median) / np.maximum(actual_last_4, 1))) * 100
                    total_mape += mape
                    
                    # Check if forecast was within 80% confidence interval
                    forecast_lower = product_forecast.iloc[:4]["Lower 80%"].values
                    forecast_upper = product_forecast.iloc[:4]["Upper 80%"].values
                    
                    within_ci = np.all((actual_last_4 >= forecast_lower) & (actual_last_4 <= forecast_upper))
                    if within_ci:
                        accurate_forecasts += 1
        
        avg_mape = total_mape / products_evaluated if products_evaluated > 0 else 0
        accuracy_rate = (accurate_forecasts / products_evaluated) * 100 if products_evaluated > 0 else 0
        
        return {
            "products_evaluated": products_evaluated,
            "average_mape": round(avg_mape, 2),
            "forecasts_within_confidence_interval": accurate_forecasts,
            "accuracy_rate": round(accuracy_rate, 2),
            "forecast_quality": "Excellent" if avg_mape < 10 else "Good" if avg_mape < 20 else "Needs Improvement"
        }
        
    except Exception as e:
        print(f"Error in forecast accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/roi-summary")
async def calculate_roi_summary():
    """Calculate overall ROI and business impact"""
    try:
        # Get all component metrics
        stockout_metrics = await calculate_stockout_prevention()
        inventory_metrics = await calculate_inventory_efficiency()
        forecast_metrics = await calculate_forecast_accuracy()
        
        # Calculate total monthly savings
        monthly_revenue_protected = stockout_metrics["potential_lost_revenue_prevented"]
        monthly_carrying_cost_saved = inventory_metrics["monthly_carrying_cost_saved"]
        
        # Assume emergency ordering costs 40% more than planned ordering
        # Estimate 2 emergency orders per month prevented per at-risk product
        emergency_orders_prevented = stockout_metrics["products_at_risk"] * 2
        avg_emergency_order_cost = 100  # Extra cost per emergency order (still estimated)
        monthly_emergency_cost_saved = emergency_orders_prevented * avg_emergency_order_cost
        
        total_monthly_savings = (
            monthly_revenue_protected + 
            monthly_carrying_cost_saved + 
            monthly_emergency_cost_saved
        )
        
        total_annual_savings = total_monthly_savings * 12
        
        return {
            "monthly_impact": {
                "revenue_protected": round(monthly_revenue_protected, 2),
                "carrying_cost_saved": round(monthly_carrying_cost_saved, 2),
                "emergency_order_cost_saved": round(monthly_emergency_cost_saved, 2),
                "total_monthly_savings": round(total_monthly_savings, 2)
            },
            "annual_impact": {
                "total_annual_savings": round(total_annual_savings, 2)
            },
            "operational_improvements": {
                "stockout_reduction": f"{stockout_metrics['improvement_percentage']}%",
                "inventory_optimization": f"{inventory_metrics['optimal_rate']}% of products optimally stocked",
                "forecast_accuracy": f"{forecast_metrics['accuracy_rate']}% accuracy rate"
            },
            "key_metrics": {
                "products_tracked": stockout_metrics["total_products_tracked"],
                "stockouts_prevented": stockout_metrics["stockouts_prevented"],
                "excess_inventory_reduced": round(inventory_metrics["excess_inventory_units"], 2),
                "forecast_quality": forecast_metrics["forecast_quality"]
            }
        }
        
    except Exception as e:
        print(f"Error in ROI summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/dashboard-summary")
async def get_dashboard_summary():
    """Get a concise summary for dashboard display"""
    try:
        roi = await calculate_roi_summary()
        
        return {
            "headline_metrics": {
                "monthly_savings": f"${roi['monthly_impact']['total_monthly_savings']:,.0f}",
                "annual_savings": f"${roi['annual_impact']['total_annual_savings']:,.0f}",
                "stockouts_prevented": roi['key_metrics']['stockouts_prevented'],
                "products_optimized": roi['key_metrics']['products_tracked']
            },
            "value_propositions": [
                {
                    "title": "Revenue Protection",
                    "value": f"${roi['monthly_impact']['revenue_protected']:,.0f}/month",
                    "description": "Prevented lost sales from stockouts"
                },
                {
                    "title": "Cost Reduction",
                    "value": f"${roi['monthly_impact']['carrying_cost_saved']:,.0f}/month",
                    "description": "Saved on inventory carrying costs"
                },
                {
                    "title": "Operational Efficiency",
                    "value": f"${roi['monthly_impact']['emergency_order_cost_saved']:,.0f}/month",
                    "description": "Eliminated emergency ordering costs"
                }
            ]
        }
        
    except Exception as e:
        print(f"Error in dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/assumptions")
async def get_metric_assumptions():
    """Show the business assumptions used in calculations"""
    return {
        "pricing_data": {
            "source": "Product_Lookup table - ProductPrice column",
            "fallback": "If price missing, uses $50 default"
        },
        "cost_data": {
            "source": "Product_Lookup table - ProductCost column",
            "estimation": "If cost missing, estimates as 60% of ProductPrice",
            "fallback": "If both missing, uses $30 default"
        },
        "cost_assumptions": {
            "annual_carrying_cost_rate": 0.25,
            "description": "25% of inventory value per year (industry standard)"
        },
        "operational_assumptions": {
            "emergency_orders_per_at_risk_product": 2,
            "extra_cost_per_emergency_order": 100,
            "description": "Based on industry averages for rush orders"
        },
        "note": "Revenue and inventory costs now use actual product prices from your database!"
    }