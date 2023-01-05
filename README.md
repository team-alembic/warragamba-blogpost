# Warragamba Blogpost

This is the accompanying repository for the post.

---

To run the jupyter notebook:

```
# I've verified this works on python 3.10.9
# I'm using asdf as the python version manager

# install correct version of Python, Elixir and Erlang
asdf install

# installs dependencies
poetry install

# starts the shell
poetry shell

# start notebook with
jupyter notebook
```

If using VSCode don't run the last command, use the Jupyter notebooks extension and use the Python env denoted as 'Poetry env'

---

To run the livemd you must install [Elixir Livebook](https://livebook.dev/) Then open it from the livebook UI.

My instance of livebook was running on:

```
erlang 25.1.2
elixir 1.14.2-otp-25
```
