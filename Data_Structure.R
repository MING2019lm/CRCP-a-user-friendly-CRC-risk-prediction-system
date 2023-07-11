data_path <- file.path("result", "mm.csv")

install.packages("ggplot2")
install.packages("reshape2")
install.packages("magick")
library(magick)
library(ggplot2)
library(reshape2)

mm <- read.csv(data_path)
mm_melt <- melt(mm,id='label')
group <- factor(mm_melt$label)
p <- ggplot(mm_melt) +
  geom_boxplot(aes(x=variable, y=value, fill=group), outlier.size = 0.3) +
  theme_bw() +
  scale_fill_manual(labels=c("PE", "CRC"), values=c("#0B9DFF", "#F56700")) +
  labs(x="", y="", fill="") +
  theme(axis.text.x = element_text(angle=45, family="serif", vjust=0.6, size=12), 
        axis.text.y = element_text(family="serif", size=12),
        axis.title.x = element_text(family="serif", size=16),
        legend.text = element_text(family="serif", size=12),
        legend.title = element_text(family="serif", size=12))
data_path1 <- file.path("result", "Data_structure.png")
ggsave(data_path1, plot = p, width = 10, height = 5, dpi = 1200)



