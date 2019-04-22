# these are advise version

for mb in 16 32 64 128
do
	for resolution in 56 112 224 448
	do
		for cpy in 0 1
		do
			for adv in 0 1
			do
				for gpun in 1 2 4
				do
					./memSwap.exe  resnet 100 ${mb} ${resolution} ${cpy} 1 ${adv}  ${gpun}
				done
			done
		done
	done
done

#./memSwap.exe  resnet 100 16  56  0 1 1 
#./memSwap.exe  resnet 100 16  56  1 1 1
                                                   
