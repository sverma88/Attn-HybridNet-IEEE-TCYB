function y = complbp(x, patchsize, mapping)

y = lbp(reshape(x, patchsize, patchsize),1,8,mapping,'h');

end