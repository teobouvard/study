.section .init9

main:
	/* initialisation de la diode */
	mov.b #32, &26
	mov   #32, r15
start:
	mov   #227, r13
	mov   #800, r11
	call #init_son
	
	mov   #1, r13
	mov   #2000, r11
	call #init_son
	
	mov   #227, r13
	mov   #800, r11
	call #init_son
	
	mov   #1, r13
	mov   #2000, r11
	call #init_son
	
	mov   #227, r13
	mov   #800, r11
	call #init_son
	
	mov   #287, r13
	mov   #560, r11
	call #init_son
	
	mov   #191, r13
	mov   #240, r11
	call #init_son
	
	mov   #227, r13
	mov   #800, r11
	call #init_son
	
	mov   #287, r13
	mov   #560, r11
	call #init_son
	
	mov   #191, r13
	mov   #240, r11
	call #init_son
	
	mov   #227, r13
	mov   #1040, r11
	call #init_son	
	
	mov   #1, r13
	mov   #50000, r11
	call #init_son
	
	mov   #152, r13
	mov   #800, r11
	call #init_son	
	
	mov   #1, r13
	mov   #2000, r11
	call #init_son	
	
	mov   #152, r13
	mov   #800, r11
	call #init_son	
	
	mov   #1, r13
	mov   #2000, r11
	call #init_son	
	
	mov   #152, r13
	mov   #800, r11
	call #init_son	
	
	mov   #143, r13
	mov   #560, r11
	call #init_son	
	
	mov   #191, r13
	mov   #240, r11
	call #init_son
	
	mov   #241, r13
	mov   #800, r11
	call #init_son
	
	mov   #287, r13
	mov   #560, r11
	call #init_son
	
	mov   #191, r13
	mov   #240, r11
	call #init_son
	
	mov   #227, r13
	mov   #1040, r11
	call #init_son	
	
	mov   #1, r13
	mov   #50000, r11
	call #init_son
	jmp start	

init_son:
	mov   #0, r14
	mov   #0, r12
son:
	mov.b r15, &25
	xor #32, r15
	add #1, r12
tone:
	add #1, r14
	cmp r14, r13
	jnz tone
	mov #0, r14		

	cmp r12, r11
	jz return
	jmp son
return:
	ret
