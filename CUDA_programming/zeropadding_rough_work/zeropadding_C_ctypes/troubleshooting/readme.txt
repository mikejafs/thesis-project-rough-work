Used this folder to diagnose the segmentation fault issue.

The solution was to be sure I was defining the intermediary vairables
to have the same dtypes as the ititial ones. In this case I just had to set 
everything to long essentially. 