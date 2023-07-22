function pro = getProbability(score)    
pro = 1./(1+exp(-abs(score)));
end  


