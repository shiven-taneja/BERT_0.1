require(reshape2) 
df <- read.csv("G:/My Drive/University of Toronto/Senior Year/Winter Semester/COG403/Final Project/data.csv", header=T)
df2 <- df[,c('t.f', 'model', 'score')]
df2$t.f <- as.logical(df2$t.f)
require(ggplot2)
# 
# p <- ggplot(data = df2, aes(x=model, y=score))
# p <- p + geom_boxplot(aes(fill = t.f))
# # if you want color for points replace group with colour=Label
# p <- p + geom_point(aes(y=score, group=t.f), position = position_dodge(width=0.75))
# p <- p + facet_wrap( ~ model, scales="free")
# p <- p + guides(fill=guide_legend(title="Legend"))
# p
df1 <- df
df1$t.f <- as.logical(df1$t.f)
# df1 <-subset(df, grammar == 'prov')
labs=c('art' = 'Articles', 'opps' = 'Opposites', 
       'prep' = 'Prepositions', 'prov' = 'Proverbs', 'quest'='Question Tags')
p <- ggplot(df1, aes(x=t.f, y=score))
p <- p + xlab("") + ylab("Score") 
p + geom_point(aes(colour = model), size = 3.5) + facet_grid(cols = vars(grammar), labeller = as_labeller(labs))


