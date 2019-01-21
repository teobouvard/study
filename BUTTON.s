.section .init9

main:
	/* initialisation de la diode */
	mov.b #2, &50
	mov   #2, r15
	mov   #3, r14
	mov.b #0, &34
	mov.b #0, &49
loop:
comp1:	
	cmp &32, r14
	jeq comp1
comp2:
	cmp &32, r14
	jne comp2

	mov.b r15, &49
	xor #2, r15
	
	jmp loop
