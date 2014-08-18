%% rectified Linear Unit gradient function

function g = reluPrime(z)

    g = (z > 0);

end
