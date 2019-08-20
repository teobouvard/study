.section .init9

main:
	/* initialisation de la diode */
	mov.b #2, &50 /*P5DIR*/
	mov   #2, r15 /*reg pour xor led*/
	mov   #0, r14 /*reg pour compteur wait*/
	mov   #50000, r13 /*reg pour max wait*/
	mov.b #0, &34 /*P1DIR*/
	mov   #3, r12 /*reg pour bouton lever*/
	mov   #0, r11 /*compteur boucle wait*/

loop:
comp1:	
	mov.b &32, r10
	cmp r10, r12
	jeq comp1
comp2:
	mov.b &32, r10
	call #wait
	add #1, r11
	cmp r10, r12
	jnz comp2
	
	add r11, r11
	call #blinkn
	jmp loop

blinkn:
	call #wait
	mov.b r15, &49 /*P5OUT*/
	xor #2, r15
	sub #1, r11
	jnz blinkn
	ret

wait:
	add #1, r14
	cmp r14, r13
	jnz wait
	mov   #0, r14
	ret
