import math

# Given points
t1 = 0
threshold1 = 0.90

t2 = 100
threshold2 = 0.10

# The function is threshold = A * exp(k * t)

# Step 1: Use the first point (t=0) to find A
# At t=0, threshold = A * exp(k * 0) = A * exp(0) = A * 1 = A
A = threshold1

# Step 2: Use the second point (t=100) and the calculated A to find k
# threshold2 = A * exp(k * t2)
# threshold2 / A = exp(k * t2)
# ln(threshold2 / A) = k * t2
# k = ln(threshold2 / A) / t2

if t2 == t1:
    print("Error: The two time points are the same. Cannot determine k.")
elif threshold1 == 0 or threshold2 == 0:
    # Technically ln(0) is undefined, handle cases where threshold might be 0
    # For decay, threshold should generally be positive
     print("Error: Threshold values must be non-zero for log calculation (or one is 0 at infinity).")
     # Could add handling for threshold2=0 meaning k goes to -infinity?
     # But for threshold2 > 0, proceed:
     if threshold2 / A <= 0:
          print("Error: Ratio threshold2 / A is non-positive. Cannot compute natural logarithm.")
else:
    try:
        ratio = threshold2 / A
        k = math.log(ratio) / t2

        # Print the results
        print(f"Given points: ({t1}, {threshold1}) and ({t2}, {threshold2})")
        print(f"The exponential decay function is of the form: threshold = A * exp(k * t)")
        print(f"Calculated parameters:")
        print(f"  A = {A}")
        print(f"  k = {k}")

        # --- Verification ---
        print("\nVerification:")
        calculated_threshold1 = A * math.exp(k * t1)
        print(f"  At t = {t1}: Calculated threshold = {calculated_threshold1:.4f} (Expected: {threshold1})")

        calculated_threshold2 = A * math.exp(k * t2)
        print(f"  At t = {t2}: Calculated threshold = {calculated_threshold2:.4f} (Expected: {threshold2})")

    except ValueError as e:
        print(f"Error during calculation: {e}")
        print("Check input values. Logarithm requires a positive argument.")