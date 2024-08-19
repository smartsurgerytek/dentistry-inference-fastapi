# Coding Convention

In general, we follow the Google coding style guide.

One can track the documents from:
https://google.github.io/styleguide/pyguide.html



For the sake of convenience, the minimum requirement is a docstring in the main functions, i.e., the nodes of the topological sort order.

Additionally, the class is NOT necessary, but ensure a well-structured arrangement of folders, as we are still in the early stages of development.

Last, kindly provide your demonstrations in notebook folder.

## Docstring
```
  def connect_to_next_port(self, minimum: int) -> int:
    """Connects to the next available port.

    Args:
      minimum: A port value greater or equal to 1024.

    Returns:
      The new minimum port.

    Raises:
      ConnectionError: If no available port is found.
    """
    ......
```

## Conditional Expressions
If the condition over two or too long, be sure to split it into lines. 
```
    one_line = 'yes' if predicate(value) else 'no'
    slightly_split = ('yes' if predicate(value)
                      else 'no, nein, nyet')
    the_longest_ternary_style_that_can_be_done = (
        'yes, true, affirmative, confirmed, correct'
        if predicate(value)
        else 'no, false, negative, nay')
```


## Nested IF-ElSE Using Not Probably
```
def apply_coupon(user, cart):
    if user.is_first_purchase:
        if cart.total_amount >= 100:
            cart.apply_discount(10)
            print("A coupon has been applied to your purchase.")
        else:
            print("Your purchase does not meet the minimum amount for a coupon.")
    else:
        print("Coupons are only available for first-time purchases.")
    return cart
```
Early return: is better than nested if-else.
```
def apply_coupon(user, cart):

    if not user.is_first_purchase:
        print("Coupons are only available for first-time purchases.")
        return cart

    if cart.total_amount < 100:
        print("Your purchase does not meet the minimum amount for a coupon.")
        return cart

    cart.apply_discount(10)
    print("A coupon has been applied to your purchase.")
    return cart
```

