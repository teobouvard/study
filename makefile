all:
	msp430-gcc -mmcu=msp430fg4618 -mdisable-watchdog -o prog.elf exSon.s
