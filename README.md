# Circle of fifths  and Sierpiński triangle shape algorithm:

https://youtube.com/shorts/A7xMJ639gAw?si=vr9JzHWUt7zdFeHt

0. Draw the circle of fifths as an HSV colored wheel with M=12 slots using M colors and add M labels for the notes. radius=1
  ['A','D','G','C','F','Bb','Eb','Ab','Db','Gb','B','E'] is mapped to an index k=[0,1,2,3,4,5,6,7,8,9,10,11]
  The position in the complex plane for the circle of fifths and its 12 notes is given by 
  $$S_k=e^{i\frac{2\pi}{M}k}$$
  
  


  

1. Randomly choose a point inside a unitary circle - Express it as a complex number 
    
    $$Z_o=r_o e^{\phi_o}$$ 
    
    There are many ways to select a random point inside a unitary circle, 
    we try selecting a radius and and an angle, as if we where doing it with
    a compass.   
    
    $$r_{min}<r_o<r_{max}$$ 
    
    and 
    
    $$-\pi<\phi_o<\pi$$

    With a resolution=1000 points per interval  

    

2. Randomly choose between this three notes $(A_b,E,C) or (7,11,3) $ note in the circle of fifths 
   and express it as a complex number $Z_{note}=S_k$ using the mapping given by the index
    
    $$S_k=e^{i\frac{2\pi}{M}k}$$ 
    
    always expressing the notes in this order:
    
    ['A','D','G','C','F','Bb','Eb','Ab','Db','Gb','B','E']

    
    $$0<k<M-1$$
    $$k=[0,1,2,3,4,5,6,7,8,9,10,11]$$

    * A_b is k=7

    * E is k=11

    * C is k=3


3. Draw a line between $Z_o$ and $Z_{note}$

4. Find the mid point between $Z_o$ and $Z_{note}$ and call it $Z_{\frac{1}{2}}$

5. Set $Z_o=Z_{\frac{1}{2}}$ and repeat $N_{iter}$ times from step 2 using this new value of $Z_o$ in step 3


