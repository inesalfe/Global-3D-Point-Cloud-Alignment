import matplotlib.pyplot as plt
import numpy as np

name = "Figures/fig"

fig_it = 1

fig = plt.figure(figsize=(5, 5))

ax = fig.add_subplot(1, 1, 1)
circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
ax.add_patch(circ)

# plt.axis('equal')

plt.ylim((-1.5, 1.5))
plt.xlim((-1.5, 1.5))
plt.grid()
plt.show()

fig_name = name + "_%i.png" % fig_it
fig.savefig(fig_name + '', format='png')
fig_it = fig_it + 1

it = 7

angles = [0];
new_angles = [];
old_angles = [];
rang = np.pi;

for i in range(it):

	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(1, 1, 1)
	circ = plt.Circle((0, 0), radius=1, edgecolor='blue', facecolor='None')
	ax.add_patch(circ)
	plt.ylim((-1.5, 1.5))
	plt.xlim((-1.5, 1.5))
	plt.grid()
	rang /= 2
	for ang in old_angles:
		plt.scatter(np.cos(ang), np.sin(ang), color='grey')
	if i != it-1:
		for ang in angles:
			plt.scatter(np.cos(ang), np.sin(ang), color='red')
			old_angles.append(ang)
			new_angles.append(ang+rang)
			new_angles.append(ang-rang)
	angles.clear()
	angles = new_angles.copy()
	new_angles.clear();
	plt.show()
	fig_name = name + "_%i.png" % fig_it
	fig.savefig(fig_name + '', format='png')
	fig_it = fig_it + 1
	

