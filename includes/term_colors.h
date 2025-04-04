#ifndef c07e0f_TERM_COLORS
#define c07e0f_TERM_COLORS

#define C_RESET		"\x1b[0m"

#define FG_RED		"\x1b[31m"
#define FG_GREEN	"\x1b[32m"
#define FG_YELLOW	"\x1b[33m"
#define FG_BLUE		"\x1b[34m"
#define FG_MAGENTA	"\x1b[35m"
#define FG_CYAN		"\x1b[36m"
#define FG_GRAY		"\x1b[37m"
#define FG_BRIGHT	"\x1b[1m"

#define BG_RED		"\x1b[41m"
#define BG_GREEN	"\x1b[42m"
#define BG_YELLOW	"\x1b[43m"
#define BG_BLUE		"\x1b[44m"
#define BG_MAGENTA	"\x1b[45m"
#define BG_CYAN		"\x1b[46m"
#define BG_GRAY		"\x1b[47m"

#define BG_RED_B		"\x1b[101m"
#define BG_GREEN_B		"\x1b[102m"
#define BG_YELLOW_B		"\x1b[103m"
#define BG_BLUE_B		"\x1b[104m"
#define BG_MAGENTA_B	"\x1b[105m"
#define BG_CYAN_B		"\x1b[106m"


enum ansi_color {
	RESET	= 0,
	NONE	= 1,
	BLACK	= 30,
	RED		= 31,
	GREEN	= 32,
	YELLOW	= 33,
	BLUE	= 34,
	PURPLE	= 35,
	CYAN	= 36,
	WHITE	= 37
};

enum ansi_brightness {
	FG = 1, BG = 2, FG_BG = 3
};


char* color(enum ansi_color color, enum ansi_color background, enum ansi_brightness brightness);


#endif
