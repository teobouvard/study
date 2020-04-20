# FAQ for autograder and github

Question: I do not see any scores updating in my autograder results

Answer: Make sure you have no compilation errors. It is important that you do not modify the methods and interfaces provided with the assignments. If you delete the methods and write your own code and it of course compiles locally but autograder expects the class to contain certain methods so do not modify or delete whats already there. You can just add code to the skeleton you get.

___
Question:
My autograder build is not updating.

Answer:
Try committing the changes to github account and autograder should build automatically. If it is still not updating report the issue to TA or the teacher and continue solving the assignment without autograder and commit your solutions.
____
Question:
- How to use git

Answer:
- see [submission instructions](submission.md)
  
___
Question:
- Git failure to push to remote repository
  - Error message:

To https://github.com/uis-dat550-spring20/username-labs  
! [rejected] master -> master (fetch first)  
error: failed to push some refs to 'https://github.com/uis-dat550-spring20/username-labs'  
hint: Updates were rejected because the remote contains work that you do  
hint: not have locally. This is usually caused by another repository pushing  
hint: to the same ref. You may want to first integrate the remote changes  
hint: (e.g., 'git pull ...') before pushing again.  
hint: See the 'Note about fast-forwards' in 'git push --help' for details.  

Answer:  
You won't really be able to fix this so you need to work around this problem if it occurs.
Follow these steps:
1. When in the assignements directory (using command line) go one level up:
  - cd ..
2. Now clone your private repository:
  - git clone https://github.com/uis-dat550-spring20/username-labs
3. Now use the file explorer to copy the assignments from the master repository into the private repository you just cloned.
4. From now on work directly in your private repository and use this command instead:
  - git push
  - You no longer have to use a remote when you push
5. When a new assignment is realeased go to the master assignment repository and use type in:
  - git pull origin master
6. Now copy the new assignment folder into your private repository and work on it there
-------
Question: I get the error "fatal: refusing to merge unrelated histories" when I try to pull the repository what do I do?

It happens mostly because you made commits on the github website before doing an initial commit. To resolve it run the following commands.



git pull --allow-unrelated-histories origin master

then followed by 

git push --set-upstream origin master

Note that if you created a remote called *labs* instead of origin then you should use 

git pull --allow-unrelated-histories labs master

then followed by 

git push --set-upstream labs master

