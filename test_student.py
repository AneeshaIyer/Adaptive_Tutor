from sim.student import Student

s = Student()

print("Initial mastery:", s.mastery)
print("Initial engagement:", s.engagement)

result = s.respond(1)

print("Answered correctly?", result)
print("New mastery:", s.mastery)
print("New engagement:", s.engagement)