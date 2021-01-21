package it.sf.ist.cardet;

import javax.swing.*;
import java.util.concurrent.Executors;


public class Run {
    private static JFrame mainFrame = new JFrame();
    public static void main(String[] args) throws Exception {

        ProgressBar progressBar = new ProgressBar(mainFrame, true);
        progressBar.showProgressBar("Loading model this make take several seconds!");
        UI ui = new UI();
        Executors.newCachedThreadPool().submit(()->{
            try {
                TinyYoloPrediction.getINSTANCE();
                ui.initUI();
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                progressBar.setVisible(false);
                mainFrame.dispose();
            }
        });
    }
}
