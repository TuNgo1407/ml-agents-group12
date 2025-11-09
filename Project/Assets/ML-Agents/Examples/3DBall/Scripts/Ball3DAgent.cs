using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.IO;
using System.Text;
using System.Linq;
using Random = UnityEngine.Random;

public class Ball3DAgent : Agent
{
    [Header("Specific to Ball3D")]
    public GameObject ball;
    [Tooltip("Whether to use vector observation. This option should be checked " +
        "in 3DBall scene, and unchecked in Visual3DBall scene. ")]
    public bool useVecObs;
    Rigidbody m_BallRb;
    EnvironmentParameters m_ResetParams;

    // Episode logging variables
    private float episodeCumulativeReward = 0f;
    private int episodeSteps = 0;
    private static int globalEpisodeCounter = 0;
    private static bool headerWritten = false;
    private static string logPath;
    private static string runId;

    public override void Initialize()
    {
        m_BallRb = ball.GetComponent<Rigidbody>();
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        SetResetParameters();

        // Get run ID
        runId = GetRunId();
        
        // Get the repo root directory (ml-agents-group12) 
        string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../.."));
        string resultsFolder = Path.Combine(projectRoot, "results");
        
        // Find or create the specific run folder
        string runFolder = FindOrCreateRunFolder(resultsFolder, runId);
        
        // Create CSV file path inside the run folder
        logPath = Path.Combine(runFolder, $"{runId}.csv");

        // Initialize CSV file with header
        if (!headerWritten && !File.Exists(logPath))
        {
            File.WriteAllText(logPath, "Episode_Each_Training_Session,Total_Reward,Steps\n", Encoding.UTF8);
            headerWritten = true;
            Debug.Log($"Logging episodes for run '{runId}' to: {logPath}");
        }
    }

    private string FindOrCreateRunFolder(string resultsFolder, string runId)
    {
        // Look for existing run folder (ML-Agents creates folders like: results/3DBall_1234567890/)
        var existingFolders = Directory.GetDirectories(resultsFolder)
            .Where(dir => Path.GetFileName(dir).StartsWith(runId))
            .ToArray();

        string runFolder;
        
        if (existingFolders.Length > 0)
        {
            // Use the most recent folder with this run ID
            runFolder = existingFolders.OrderByDescending(dir => Directory.GetCreationTime(dir)).First();
            Debug.Log($"Found existing run folder: {runFolder}");
        }
        else
        {
            // Create a new folder for this run
            // ML-Agents pattern: BehaviorName_Timestamp
            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            runFolder = Path.Combine(resultsFolder, $"{runId}_{timestamp}");
            Directory.CreateDirectory(runFolder);
            Debug.Log($"Created new run folder: {runFolder}");
        }
        
        return runFolder;
    }

    private string GetRunId()
    {
        // Method 1: Try to get from ML-Agents Communicator (most reliable)
        try
        {
            // ML-Agents stores the run ID in the Academy
            var academy = Academy.Instance;
            // Try to access through reflection if needed
            var academyType = academy.GetType();
            var behaviorNameField = academyType.GetField("m_BehaviorName", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (behaviorNameField != null)
            {
                string behaviorName = (string)behaviorNameField.GetValue(academy);
                if (!string.IsNullOrEmpty(behaviorName))
                    return behaviorName;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Failed to get run ID from Academy: {e.Message}");
        }

        // Method 2: Try environment variables that ML-Agents might set
        string[] envVars = {
            "MLAGENTS_RUN_ID",
            "MLAGENTS_BEHAVIOR_NAME", 
            "MLAGENTS_ENVIRONMENT_ID"
        };
        
        foreach (string envVar in envVars)
        {
            string envValue = System.Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrEmpty(envValue))
            {
                Debug.Log($"Found run ID from environment variable {envVar}: {envValue}");
                return envValue;
            }
        }

        // Method 3: Look for ML-Agents results folder pattern
        string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../.."));
        string resultsParent = Path.Combine(projectRoot, "results");
        
        if (Directory.Exists(resultsParent))
        {
            var dirs = Directory.GetDirectories(resultsParent);
            if (dirs.Length > 0)
            {
                // Get the most recently created directory (likely the current run)
                var latestDir = dirs.OrderByDescending(d => Directory.GetCreationTime(d)).First();
                string folderName = Path.GetFileName(latestDir);
                // Extract just the run ID part (before the timestamp)
                string[] parts = folderName.Split('_');
                if (parts.Length > 0)
                {
                    Debug.Log($"Using latest results folder as run ID: {parts[0]}");
                    return parts[0];
                }
                Debug.Log($"Using latest results folder as run ID: {folderName}");
                return folderName;
            }
        }

        // Method 4: Final fallback - use timestamp
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        Debug.LogWarning($"Run ID not found. Using timestamp: {timestamp}");
        return timestamp;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (useVecObs)
        {
            sensor.AddObservation(gameObject.transform.rotation.z);
            sensor.AddObservation(gameObject.transform.rotation.x);
            sensor.AddObservation(ball.transform.position - gameObject.transform.position);
            sensor.AddObservation(m_BallRb.velocity);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        episodeSteps++;

        var actionZ = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
        var actionX = 2f * Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);

        if ((gameObject.transform.rotation.z < 0.25f && actionZ > 0f) ||
            (gameObject.transform.rotation.z > -0.25f && actionZ < 0f))
        {
            gameObject.transform.Rotate(new Vector3(0, 0, 1), actionZ);
        }

        if ((gameObject.transform.rotation.x < 0.25f && actionX > 0f) ||
            (gameObject.transform.rotation.x > -0.25f && actionX < 0f))
        {
            gameObject.transform.Rotate(new Vector3(1, 0, 0), actionX);
        }
        if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        {
            SetReward(-1f);
            episodeCumulativeReward += -1f;
            EndEpisode();
        }
        else
        {
            SetReward(0.1f);
            episodeCumulativeReward += 0.1f;
        }
    }

    public override void OnEpisodeBegin()
    {
        // Log the completed episode data BEFORE resetting counters
        if (episodeSteps > 0)
        {
            globalEpisodeCounter++;
            string logEntry = $"{globalEpisodeCounter},{episodeCumulativeReward:F2},{episodeSteps}\n";
            File.AppendAllText(logPath, logEntry, Encoding.UTF8);
        }

        // Reset counters for the new episode
        episodeCumulativeReward = 0f;
        episodeSteps = 0;

        // Reset environment state
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
        m_BallRb.velocity = new Vector3(0f, 0f, 0f);
        ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f))
            + gameObject.transform.position;
        SetResetParameters();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = -Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }

    public void SetBall()
    {
        m_BallRb.mass = m_ResetParams.GetWithDefault("mass", 1.0f);
        var scale = m_ResetParams.GetWithDefault("scale", 1.0f);
        ball.transform.localScale = new Vector3(scale, scale, scale);
    }

    public void SetResetParameters()
    {
        SetBall();
    }
}