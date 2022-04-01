

def get_str(num) -> str:
	"""Returns a 4 digits string from a number.
	
	:param num:
		An integer to be translated into a string.
		
	:return:
		The string representing the number.
	:rtype str:
	"""
	if num < 10:
		return "00" + str(num)
	elif 10 <= num < 100:
		return "0" + str(num)
	else:
		return str(num)
