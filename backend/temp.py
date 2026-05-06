from database import Database

db = Database()
users = db.get_all_users()

print("USER COUNT:", len(users))
for user in users:
    print(user.id, user.name, user.role)