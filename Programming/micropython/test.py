
tmp = 0b00000000

print(bin(tmp))

tmp |= (1<<3)

print(bin(tmp))

tmp &= ~(1<<3)

print(bin(tmp))
