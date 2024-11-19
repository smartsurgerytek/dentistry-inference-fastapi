# Coding Convention

In general, we follow the Google coding style guide.

One can track the documents from:
https://google.github.io/styleguide/pyguide.html


For the sake of convenience, the minimum requirement is a docstring in the main functions, i.e., the nodes of the topological sort order.

Additionally, the class is NOT necessary, but ensure a well-structured arrangement of folders, as we are still in the early stages of development.

Last, kindly provide your demonstrations in the notebooks folder.

Let's follow the github workflow with feature branch:
## Workflow
Before starting to work on a new feature or fix, create a new branch off the main branch (or another base branch like develop). This keeps your changes isolated from the main code.
### Create future Branch

```
git fetch # get the current branches
git checkout main         # Make sure you're on the main branch
git pull origin main      # Pull the latest changes from the remote repository
git checkout -b feature/your-feature-name  # Create and switch to a new branch
```
### Make Your Changes
Now, make your changes to the code, such as adding a new feature, fixing bugs, or updating documentation.
```
git status
```

### Stage and Commit Your Changes
After making changes, you need to stage and commit them to your feature branch.
```
git add .  # Stage all changes (use specific file names if you want to stage specific files)
git commit -m "Add feature: description of the change"
```
### Push Your Feature Branch to GitHub
```
git push origin feature/your-feature-name
```
### Create a Pull Request (PR)
Now that your feature branch is on GitHub, you can create a Pull Request (PR) to propose merging your feature branch into the main branch (or whichever branch your team uses as the base branch, like develop). 

Press Pull requests and goes to New pull request
![img1](/docs/PR_page1.png)
Choose the source (feature branch) and target branch (Main)

Then create PR
![img2](/docs/PR_page2.png)

Kindly tag me in line or slack, I will review that ASAP!

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


## Nested IF-ElSE
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