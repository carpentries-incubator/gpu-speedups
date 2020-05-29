[![Create a Slack Account with us](https://img.shields.io/badge/Create_Slack_Account-The_Carpentries-071159.svg)](https://swc-slack-invite.herokuapp.com/)

# GPU Speedups in Python

This lesson explores how to parallelize Python code with GPU speedups. There are a number of ways to do this, and some methods overlap with others.

## Audience
The intended audience are people familiar with Python, the python library `numpy`, and that have some initial familiarity with the ideas behind parallelization in computer programming. 


## Why GPUs?
Many computational problems are parallelizable and can be sped up 10-1000 times on the GPU. Suitable computations are doing the same thing to pieces of 1000s - trillions of pieces of data (records, samples, entries, etc).

Let's think of a familiar example: the average. An average of numbers stored in a vector can be computed by accumulating these numbers in one spot, and then dividing by the total number. The average does not depend on what order they are added together in, and no number "speaks" to another number. A for loop could do this, but why use a loop? Every pass through the loop is independent of each other. What if all the passes through the loops could be done at the same time? If you have problems like these, GPU speedups can pay off.

Python already has libraries for vectorized problems, like numpy. However, much of this code is actually a low level (eg in C) implementation that is non paralleled, but just loop really fast. There is overhead transferring data to the GPU, but if the computation is complicated enough, then doing it on the GPU can be worth it.

Also, some problems are not (easily) vectorizable or written as fast Numpy ufuncs. For example, each sub computation is independent, but you need some if/else clauses and can't put this into a matrix. If you cannot write down the problem as a series of matrix operations, but can write out what to do in each case, then you can code a kernel function that the GPU cores execute. There are 1000s of GPU cores on a single GPU device, unlike dozens of CPUs.

## Contributing

We welcome all contributions to improve the lesson! Maintainers will do their best to help you if you have any
questions, concerns, or experience any difficulties along the way.

We'd like to ask you to familiarize yourself with our [Contribution Guide](CONTRIBUTING.md) and have a look at
the [more detailed guidelines][lesson-example] on proper formatting, ways to render the lesson locally, and even
how to write new episodes.

Please see the current list of [issues][FIXME] for ideas for contributing to this
repository. For making your contribution, we use the GitHub flow, which is
nicely explained in the chapter [Contributing to a Project](http://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project) in Pro Git
by Scott Chacon.
Look for the tag ![good_first_issue](https://img.shields.io/badge/-good%20first%20issue-gold.svg). This indicates that the maintainers will welcome a pull request fixing this issue.  


## Maintainer(s)

Current maintainers of this lesson are 

* Geoffrey Woollard (@geoffwoollard)


## Authors

A list of contributors to the lesson can be found in [AUTHORS](AUTHORS)

## Citation

To cite this lesson, please consult with [CITATION](CITATION)

[lesson-example]: https://carpentries.github.io/lesson-example
