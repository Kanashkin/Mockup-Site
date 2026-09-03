"""Run this ONCE (locally or via `railway run python setup_paypal.py`) after
PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET are set, to create the PayPal
product+plan the app will sell subscriptions against. It prints the plan id
you then set as PAYPAL_PLAN_ID.

Usage:
    PAYPAL_CLIENT_ID=... PAYPAL_CLIENT_SECRET=... PAYPAL_MODE=sandbox \
        python setup_paypal.py [price_usd] [interval_unit]

price_usd defaults to 9.99, interval_unit defaults to MONTH (or YEAR).
"""
import sys
import paypal

if not paypal.configured():
    print("Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET first.")
    sys.exit(1)

price = sys.argv[1] if len(sys.argv) > 1 else "9.99"
interval = sys.argv[2] if len(sys.argv) > 2 else "MONTH"

print(f"Mode: {paypal.PAYPAL_MODE}  Price: ${price}/{interval.lower()}")

product_id = paypal.create_product(
    name="T-Shirt Mockup Pro",
    description="High-resolution mockup downloads",
)
print(f"Created product: {product_id}")

plan_id = paypal.create_plan(
    product_id=product_id,
    name=f"T-Shirt Mockup Pro ({interval.title()}ly)",
    price_usd=price,
    interval_unit=interval,
)
print(f"Created plan: {plan_id}")
print()
print(f"Set this on Railway:  PAYPAL_PLAN_ID={plan_id}")
