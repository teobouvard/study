![UiS](https://www.uis.no/getfile.php/13391907/Biblioteket/Logo%20og%20veiledninger/UiS_liggende_logo_liten.png)

# Lab 1: Getting Started

| Lab 1:           | Getting Started             |
| ---------------- | --------------------------- |
| Subject:         | DAT520 Distributed Systems  |
| Deadline:        | Thursday Jan 24 2019 14:00  |
| Expected effort: | 10-15 hours                 |
| Grading:         | Pass/fail                   |
| Submission:      | Individually                |

### Table of Contents

1. [Introduction](https://github.com/uis-dat520-s2019/assignments/blob/master/lab1/README.md#introduction)
2. [Lab Overview](https://github.com/uis-dat520-s2019/assignments/blob/master/lab1/README.md#lab-overview)
3. [Prerequisites](https://github.com/uis-dat520-s2019/assignments/blob/master/lab1/README.md#prerequisites)
4. [Go Assignments](https://github.com/uis-dat520-s2019/assignments/blob/master/lab1/README.md#go-assignments)
5. [Lab Approval](https://github.com/uis-dat520-s2019/assignments/blob/master/lab1/README.md#lab-approval)

## Introduction

This first lab will give you an overview of the lab project as a whole.
Additionally, it provides information about the required prerequisites needed.
The lab also contains some basic programming tasks to get you familiar with the
Go programming language.

## Lab Overview

The lab project for this course consist of six assignments (the list may change
during the semester):

1. Getting started
2. Network programming in Go
3. Failure Detector and Leader Election
4. Single-decree Paxos
5. Multi-Paxos
6. Replicated State Machine

The first two assignments are introductory and individual. They are intended to
get you up to speed on the Go programming language and basic network
programming. The remaining assignments form the main lab project and should be
done in groups.

The aim of the overall lab project is to implement a fault tolerant
application, replicated on several machines, using the Paxos protocol. The
Paxos protocol will be introduced in the course accordingly. For this we will
use the Go programming language. Go is very well suited to implement the
event-driven structure of the Paxos protocol. You will for each lab construct
an independent part of the application, resulting in a complete implementation
in Lab 6.

## Prerequisites

This section will give an introduction to the various tools needed for the lab
project in this course. The tools are:

1. **Autograder** - A tool for submitting and getting feedback on lab
   assignments.
2. **The GNU/Linux Lab at UiS** - A lab with workstations available for
   students to run and test their implementations.
3. **Git and GitHub** - The source code management system and Git web-hosting
   service to be used.
4. **Go** - The programming language used for all lab assignments.

#### The Autograder Tool

This course uses a new tool called Autograder. It is a tool for students and
teaching staff for submitting and validating lab assignments and is developed
at the University of Stavanger. All lab submissions from students are handled
using Git, a source code management system, and GitHub, a web-based hosting
service for Git source repositories.

**Task - Autograder registration:**

You will have to sign up for the lab project in Autograder if you have not already
done so. Instructions for this can be found
[here](http://github.com/uis-dat520-s2019/course-info/autograder-registation.md).

#### The GNU/Linux Workstation Lab

In this lab project you are expected to run your code on several machines that
interact with each other. Thus, it is useful to know how this can be done. To
test and run your code on several machines you can use the GNU/Linux lab in
E353, where the machines are named: `pitter1` - `pitter40`.

You can use `ssh` to log onto another machine without physically going to that
machine and login there. This makes it easy to run and test the example code
and your project later. To log onto a machine using ssh, open a terminal window
and type ssh username@hostname, with username and hostname replaced
accordingly. For example, to log onto one of the machines in the Linux lab,
type `ssh username@pitter18.ux.uis.no`. You may omit `.ux.uis.no` from the
command if you are connecting from a machine already located on the campus Unix
network. Executing the command will prompt for your password. Enter your
account password and you will be logged into that machine remotely.

You can avoid having to type the password each time by generating a ssh
key-pair. Such a key-pair may already have been generated for your Unix
account. The key-pair should be located in `$HOME/.ssh` folder and be named
`id_rsa` (private key) and `id_rsa.pub` (public key). The `ssh-keygen` command
can be used if you need or want to generate a new public-private key-pair.
Type `man ssh-keygen` and read the instructions on how to use the tool. Then
try running this command to generate your own key-pair. For convenience, make
sure that once asked to give a password for your private key, just press enter
at the password prompt. Once the key-pair have been generated, copy the
contents of the public-key file (ends with .pub) to a file named authorized
keys. If you have multiple keys in the latter file, make sure not to overwrite
those keys, and instead paste the new public-key at the end of your current
file. After having completed this process, try to connect to another machine
using ssh and see whether you have to type the password again.

_Note:_ You may password protect your private key and avoid typing the password
for every remote login by using `ssh-agent`. The tool remembers the decrypted
private key so that you as a user do not need to type a passphrase every time.
Interested students are referred to online tutorials and documentation.

**Task - registration:**

You will need an Unix account to access machines in the GNU/Linux lab. Get an
account for UiS’ Unix system by following the instructions on
http://user.ux.uis.no.

**Task - getting to know a Unix-like system:**

If you have not worked on a Unix or GNU/Linux system before, make yourself
acquainted with the basic functionality of a Unix shell. There are many
different tutorials available online. One introduction is
http://www.ee.surrey.ac.uk/Teaching/Unix/. You will need many of the commands
mentioned there to use the remote login shell described above (ssh).

#### Git and GitHub

Basic knowledge of Git and GitHub is needed when working on the lab
assignments:

- Git is a distributed revision control and source code management system.
  Basic knowledge of Git is required for handing in the lab assignments. There
  are many resources available online for learning Git. A good book is _Pro
  Git_ by Scott Chacon. The book is available for free at
  http://git-scm.com/book. Chapter 2.1 and 2.2 should contain the necessary
  information for delivering the lab assignments.

- GitHub is a web-based hosting service for software development projects that
  use the Git revision control system. An introduction to Git and GitHub is
  available in this video: http://youtu.be/U8GBXvdmHT4. Students need to sign
  up for a GitHub account to get access to the required course material.

This course is registered as a GitHub organization. The organization can be
found at http://github.com/uis-dat520-s2019.

#### The Go Programming Language

We will be using the Go programming language throughout the lab project. Go is
a very nice language in many ways. It provides built in primitives to design
concurrent programs using lightweight threads, called goroutines. In addition,
these goroutines can communicate using channels, instead of by means of shared
memory which is the most common approach in other languages such as C and Java.
Go is also a very small language in the sense that it has very few keywords,
and as such it does not take too long to learn the fundamentals of the
language.

There are many tutorials, books and other documentation on the Go language
available on the web. When searching the web, use the keyword `golang` instead of
go. The main source of information should preferably be golang.org, which
offers extensive documentation. Another great way to learn Go is to complete
the tour linked in the task below.

##### Programming tools

`1`. [Visual Studio Code](https://code.visualstudio.com/) 

Visual Studio Code is a great cross-platform editor with support for the Go language through an extension. We recommend using this editor since it is easy to setup and has good support for debugging.

`2.` [GoLand](https://www.jetbrains.com/go/)

GoLand is another good cross platform IDE for programming Go.


`3.` [Sublime](https://www.sublimetext.com/3)

Here are some directions to get Sublime Text to work with Go. You will need to install three things.

Download and extract the 64-bit Linux tarball.

```
	cd $HOME/Downloads
	tar -C <destination directory> -xvf <tarball file name>
```

Example:

```
	tar -C $HOME -xvf sublime_text_3_build_3103_x64.tar.bz2
```

Adding the location of the `sublime_text` executable to your path in the `.bashrc` file in your `$HOME` directory can make opening sublime easier.

```
	cd $HOME
	vi .bashrc
```

Add the following line:

```
	export PATH=$PATH:<directory of sublime_text>
```

Example:

```
	export PATH=$PATH:$HOME/sublime_text_3
```

Now you should be able to open sublime just by typing
`sublime_text`

`4.` [Package Control](https://packagecontrol.io/)

Just follow the instructions [here](https://packagecontrol.io/installation).

`5.` [GoSublime](https://github.com/DisposaBoy/GoSublime)

Just follow the instructions [here](https://github.com/DisposaBoy/GoSublime#installation).

Whichever editor you choose, it is highly recommended that you configure it to
use the goimports tool. This will reformat your code to follow the Go style,
and make sure that all the necessary import statements are inserted (so you
don't need to write the import statement when you start using a new package.)
The goimports tool is compatible with most editors, but may require some
configuration.

Note that editors may also be able to run your code within the editor itself,
but it may require some configuration. However, using the go tool from a
terminal window (i.e. the command line) is often times preferred.

**Task - learn the basics of Go:**

Start learning the basics of Go if you are not familiar with the language. We
_strongly_ recommend to work through the tour at http://tour.golang.org if
you've not done so before. Note that this task is _not_ compulsory and will not
be graded. You are encouraged to complete as much of the tour as you feel
necessary. We also recommend to revisit relevant parts of the tour throughout
the course.

#### Installing and Running Go Code

The latest version of Go should be installed on all the machines in the
GNU/Linux lab. You need to follow the installation instructions found
[here](http://golang.org/doc/install) if you wish to use your own machine for
Go programming.

**Task - check your Go installation:**

Set up a workspace, try to install and run simple packages, as explained on
[here](http://golang.org/doc/code.html). Don't forget to export your workspace
as `$GOPATH`. Assuming that you have configured `$GOPATH` correctly,
you can run the go tool and its subcommands from any directory.

**_It is very important that your Go installation and workspace setup is
verified working correctly before you proceed._**

## Go Assignments

This section offers step-by-step instructions on how to complete and hand in
Lab 1. Please refer to the workflow described below also for future labs unless
otherwise noted. The tasks will introduce you to some basic programming in Go.
You may find them easy if you have previous experience with the language, but
they serve as a good example of how to work with Autograder.

1. You will have access to two repositories when you have registered using
   Autograder. The first is the `assignments` repository, which is where we will
   publish all lab assignments, skeleton code and additional information
   needed. You only have read access to this repository. The second repository
   is your own repository named `username-labs`. `username` should be
   substituted with your own GitHub username. You have write access to this
   repository. Your answers to the assignments should be pushed here.

2. To get started with the Go part of this lab, you can now use the `go get`
   command to clone the original `assignments` repository. Here is how to do it: On
   the command line enter: `go get github.com/uis-dat520-s2019/assignments` (ignore the
   message about no buildable Go files). This will clone the original `assignments`
   git repo (not your copy of it.) This is important because it means that you
   don't need to change the import path in the source files to use your own
   repository's path. That is, when you make a commit and push to submit your
   handin, you don't have to change this back to the original import path.

3. Change directory to: `cd $GOPATH/src/github.com/uis-dat520-s2019/assignments`. Next, run
   the following command: `git remote add labs https://github.com/uis-dat520-s2019/username-labs` where `username` should be
   replaced with your own GitHub username.

4. The above command adds your own `username-labs` repository as a remote
   repository on your local machine. This means that once you've modified some
   files and committed the changes locally, you can run: `git push labs` to
   have them pushed up to your own `username-labs` repository on GitHub.

5. Note to advanced users: Follow these
   [steps](https://github.com/uis-dat520-s2019/course-info/blob/master/github-ssh.md)
   if you want to use SSH for GitHub authentication.

6. If you make changes to your own `username-labs` repository using the GitHub
   web interface, and want to pull those changes down to your own computer, you
   can run the command: `git pull labs master`. In later labs, you will work in
   groups. This approach is also the way that you can download (pull) your
   group's code changes from GitHub, assuming that another group member has
   previously pushed it out to GitHub.

7. As time goes by we (the teaching staff) will be publishing updates to the
   original `assignments` repo, e.g. new lab assignments. To see these updates, you will
   need to run the following command: `git pull origin master`.

8. For the first set of labs we will provide you with skeleton code and a set of
   tests. Thus, you will have to implement the missing pieces of the skeleton code,
   and verify that your implementation passes the available tests.
   Note that Autograder will run an additional set of test cases to verify your
   implementation. Not all tests must pass to get a passing grade.

9. In the following, we will use **Task 1** as an example. Change directory to:
   `cd $GOPATH/src/github.com/uis-dat520-s2019/assignments/lab1` and confirm that the files
   for lab1 resides in that folder. They should, assuming that you ran the `go get` command earlier. The file `fib.go` contain the following skeleton code:

   ```go
   package lab1

   // Task 1: Fibonacci numbers
   //
   // fibonacci(n) returns nth Fibonacci number, and is defined by the
   // recurrence relation F_n = F_n-1 + F_n-2, with seed values F_0=0 and F_1=1.
   func fibonacci(n uint) uint {
   0
   }
   ```

10. Implement the function body according to the specification so that all the
    tests in the corresponding `fib_test.go` file passes. The file looks like
    this:

    ```go
    package lab1

    import "testing"

    var fibonacciTests = []struct {
        in, want uint
    }{
        {0, 0},
        {1, 1},
        {2, 1},
        {3, 2},
        {4, 3},
        {5, 5},
        {6, 8},
        {7, 13},
        {8, 21},
        {9, 34},
        {10, 55},
        {20, 6765},
    }

    func TestFibonacci(t *testing.T) {
        for i, ft := range fibonacciTests {
    	    out := fibonacci(ft.in)
    	    if out != ft.want {
    		    t.Errorf("fib test %d: got %d for input %d, want %d", i, out, ft.in, ft.want)
    	    }
        }
    }
    ```

11. If you run `go test` without any arguments, the tool will run all the tests
    found in files with name matching the pattern "\*\_test.go". You may only run
    a specific test by providing the `-run` flag to `go test`. For example, `go test -run TestFib` will only run the `TestFibonacci` test. Generally,
    running `go test -run regexp` will run only those tests matching the
    regular expression `regexp`.

12. You should **_not_** edit files or code that are marked with a `// DO NOT EDIT` comment. Please make separate `filename_test.go` files if you wish
    to write and run your own tests.

13. When you have completed a task and sufficiently many local tests pass, you
    may push your code to GitHub. This will trigger Autograder which will then
    run a separate test suite on your code.

14. Using the Fibonacci task (`fib.go`) as an example, use the following
    procedure to commit and push your changes to GitHub and Autograder:

    ```
    $ cd $GOPATH/src/github.com/uis-dat520-s2019/assignments/lab1
    $ git add fib.go
    $ git commit
    // This will open an editor for you to write a commit message
    // Use for example "Implemented Assignment 1"
    $ git push labs
    ```

15. Running the last command above will, due to an error on our part, result in
    Git printing an error message about a conflict between the `README.md` file
    in the `assignments` repository and the `README.md` file in your `username-labs`
    repostitory. Here is how to fix it:

    ```
    $ git push labs
    ...
    ! [rejected]        master -> master (fetch first)
    error: failed to push some refs to 'git@github.com:uis-dat520-s2019/username-labs.git'
    ...
    $ git pull labs master
    ...
    Auto-merging README.md
    CONFLICT (add/add): Merge conflict in README.md
    Automatic merge failed; fix conflicts and then commit the result.
    ...
    $ cd $GOPATH/src/github.com/uis-dat520-s2019/assignments
    $ nano README.md
    // Remove everything in the file, then add for example "username-labs" to the file.
    // Save and exit.
    $ git add README.md
    $ git commit
    $ // Use the existing (merge) commit message. Save and exit.
    $ git push labs
    // Your push should now complete successfully.
    // You may check that your changes are reflected on GitHub through the GitHub web interface.
    ```

16. Autograder will now build and run a test suite on the code you submitted.
    You can check the output by going the [Autograder web
    interface](http://autograder.ux.uis.no/). The results (build log) should be
    available under "Individual - lab1". Note that the results shows output
    for all the tests in current lab assignment. You will want to focus on the
    output for the specific test results related to the task you're working on.

17. **Task 2:** Complete the task found in `stringer.go`. You may check your
    solution locally with the tests found in `stringer_test.go`.

18. **Task 3:** Complete the task found in `rot13.go`. You may check your
    solution locally with the tests found in `rot13_test.go`.

19. **Task 4:** Complete the task found in `errors.go`. You may check your
    solution locally with the tests found in `errors_test.go`.

20. **Task 5:** Complete the task found in `multiwriter.go`. You may check your
    solution locally with the tests found in `multiwriter_test.go`.

21. When you are finished with all the tasks for the current lab, and wish
    to submit, then first make sure you commit your changes and write only the
    following: `username labX submission` in the first line of the commit
    message, where you replace `username` with your GitHub username and `X`
    with the lab number. Your usage of slip days will be calculated based on
    when you pushed this commit to GitHub. If there are any issues you want us
    to pay attention to, please add those comments after an empty line in the
    commit message. If you later find a mistake and want to resubmit, please
    use `username labX resubmission` as the commit message.

22. Push your changes using `git push labs`. You should be able to view your
    results in the Autograder web interface as described earlier.

## Lab Approval

To have your lab assignment approved, you must come to the lab during lab hours
and present your solution. This lets you present the thought process behind your
solution, and gives us a more information for grading purposes. When you are
ready to show your solution, reach out to a member of the teaching staff.
It is expected that you can explain your code and show how it works.
You may show your solution on a lab workstation or your own
computer. The results from Autograder will also be taken into consideration
when approving a lab. At least 60% of the Autograder tests should pass for the
lab to be approved. A lab needs to be approved before Autograder will provide
feedback on the next lab assignment.

Also see the [Grading and Collaboration
Policy](https://github.com/uis-dat520-s2019/course-info/policy.md)
document for additional information.
