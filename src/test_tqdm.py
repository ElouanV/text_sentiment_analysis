from tqdm import trange
from time import sleep
def get_divisors(n):
    divisors = []
    for m in range(1, n+1):
        if n % m == 0:
            divisors.append(m)
    return divisors

iterations = 10
with trange(iterations, unit="carrots") as pbar:
    for i in pbar:
        sleep(0.5)
        if i % 2:
            pbar.set_description(f"Testing odd number {i}")
        else:
            pbar.set_description(f"Testing even number {i}")
        pbar.set_postfix(divisors=get_divisors(i))