# TODO plots without a place to stay at the moment

# import seaborn as sns

# def corner_plot_sns(pandas,columns,hue,save="",**kwargs):

#     g = sns.pairplot(
#     pandas[columns],
#     hue = hue,
#     diag_kind='kde',
#     corner=True,
#     plot_kws={'alpha': 0.5, 's': 20},
#     height=5)

#     # Customize tick marks and spines
#     for ax in g.axes.flatten():
#         if ax:
#             ax.tick_params(
#                 axis='both',
#                 which='both',
#                 direction='out',
#                 length=10,
#                 width=2,
#                 labelsize=20
#             )
#             for spine in ax.spines.values():
#                 spine.set_linewidth(2)
#     legend = g._legend  # Access the legend object
#     if legend:
#         legend.set_title(hue, prop={'size': 40})  # Increase legend title size
#         for text in legend.get_texts():
#             text.set_fontsize(40)
#     if save:
#         plt.savefig(f"{save}.jpg", dpi=300, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()
