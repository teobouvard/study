package util;

/**
 *
 * @author DASI Team
 */
public class DebugLogger {

    // Méthode pour avoir des messages de Log dans le bon ordre (pause)
    public static void pause(long milliseconds) {
        try {
            Thread.sleep(milliseconds);
        } catch (InterruptedException ex) {
            ex.hashCode();
        }
    }

    // Méthode pour avoir des messages de Log dans le bon ordre (log)
    public static void log(String message) {
        System.out.flush();
        pause(5);
        System.err.println("[DebugLogger] " + message);
        System.err.flush();
        pause(5);
    }

    // Méthode pour avoir des messages de Log dans le bon ordre (log avec exception)
    public static void log(String message, Exception ex) {
        System.out.flush();
        pause(5);
        System.err.println("[DebugLogger] " + message);
        System.err.println("[**EXCEPTION**] " + ex.getMessage());
        System.err.print("[**EXCEPTION**] >>> ");
        ex.printStackTrace(System.err);
        System.err.flush();
        pause(5);
    }
    
    public static void main(String[] args) {
        
        DebugLogger.log("** Début du Test **");
        
        DebugLogger.log("Message de DEBUG pour tester...");
        
        Integer a = 0;
        Integer b = null;
        
        if (a < 0) {
            DebugLogger.log("ERREUR sur la valeur de A");
        }
        
        try {
            if (b >  0) {
                DebugLogger.log("Test OK pour la valeur de B");
            }
        } catch (Exception ex) {
            DebugLogger.log("Problème avec B", ex);
        }
        
        DebugLogger.log("** Fin du Test **");
    }
}
